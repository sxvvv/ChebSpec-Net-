import torch
import torch.nn as nn
from einops import rearrange
import math

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        
    def forward(self, x):
        # Handle 4D input correctly
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
        return x

class GroupwiseChebModule(nn.Module):
    def __init__(self, dim, K=3):
        super().__init__()
        self.K = K
        self.groups = max(2, dim // 16)  # Adaptive grouping
        self.group_dim = dim // self.groups
        
        # Enhanced weight initialization and sharing
        self.weights = nn.Parameter(torch.randn(K, self.group_dim, self.group_dim) * 0.02)
        self.scales = nn.Parameter(torch.ones(self.groups))
        
        # Efficient mixing with less parameters
        self.mix = nn.Sequential(
            nn.Conv2d(dim, dim, 1, groups=self.groups, bias=False),
            nn.GELU()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.groups, -1, H * W)
        
        outs = []
        for i in range(self.groups):
            group_x = x[:, i].transpose(-1, -2)
            prev, curr = group_x, group_x @ (self.weights[0] * self.scales[i])
            result = curr
            
            for k in range(1, self.K):
                next_out = 2 * curr @ (self.weights[k] * self.scales[i]) - prev
                prev, curr = curr, next_out
                result = result + curr * (0.8 ** k)  # Decay coefficient
            
            outs.append(result)
        
        out = torch.stack(outs, dim=1)
        out = out.transpose(-1, -2).contiguous().view(B, C, H, W)
        return self.mix(out)

class FRFN(nn.Module):
    def __init__(self, dim, hidden_dim, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        # Linear layers for dimension transformation
        self.linear1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            act_layer()
        )
        self.linear2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

        # Depthwise convolution for feature mixing
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=False),
            act_layer()
        )

    def forward(self, x):
        # Apply linear transformation to increase dimensionality
        x = self.linear1(x)

        # Apply depthwise convolution for feature mixing
        x = self.dwconv(x)

        # Apply linear transformation to restore dimensionality
        x = self.linear2(x)

        return x

class SpectroTransBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        
        self.cheb = GroupwiseChebModule(dim)
        self.temporal = SpectroTemporalGating(dim)
        
        self.transform = FRFN(dim, 4*dim)
        
    def forward(self, x, prev_state=None):
        identity = x
        x = self.cheb(self.norm1(x))
        
        temp, new_state = self.temporal(x, prev_state)
        x = identity + x + temp
        
        x = x + self.transform(self.norm2(x))
        return x, new_state

class SpectroTemporalGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        reduced_dim = max(dim // 8, 16)
        
        self.reduce = nn.Sequential(
            nn.Conv2d(dim, reduced_dim, 1),
            nn.GELU()
        )
        
        self.state_process = nn.Sequential(
            nn.Conv2d(reduced_dim * 2, reduced_dim, 1),
            LayerNorm(reduced_dim),
            nn.GELU()
        )
        
        self.gate = nn.Sequential(
            nn.Conv2d(reduced_dim, reduced_dim, 3, padding=1, groups=reduced_dim),
            nn.GELU(),
            nn.Conv2d(reduced_dim, 1, 1),
            nn.Sigmoid()
        )
        
        self.expand = nn.Conv2d(reduced_dim, dim, 1)

    def forward(self, x, prev_state=None):
        feat = self.reduce(x)
        if prev_state is None:
            prev_state = torch.zeros_like(feat)
        
        combined = self.state_process(torch.cat([feat, prev_state], dim=1))
        gate = self.gate(combined)
        new_state = gate * combined + (1 - gate) * prev_state
        
        return self.expand(new_state), combined

class SpectroTemporalNet(nn.Module):
    def __init__(self, inp_channels=3, dim=24):
        super().__init__()
        self.embed = nn.Conv2d(inp_channels, dim, 3, 1, 1)
        
        # Initialize dimensions for encoder path
        self.dims = [dim * (2**i) for i in range(3)]
        
        # Encoder modules with spatial downsampling
        self.encoders = nn.ModuleList([
            nn.ModuleDict({
                'down': nn.Conv2d(d, d*2, 4, 2, 1),
                'process': SpectroTransBlock(d*2)
            }) for d in self.dims
        ])
        
        # Middle processing
        self.middle = nn.ModuleList([
            SpectroTransBlock(self.dims[-1] * 2) for _ in range(2)
        ])
        
        # Decoder modules with spatial upsampling
        decoder_channels = [
            (self.dims[-1] * 2, self.dims[-1]),  # 192 -> 96
            (self.dims[-1], self.dims[-2]),      # 96 -> 48
            (self.dims[-2], self.dims[-3])       # 48 -> 24
        ]
        
        self.decoders = nn.ModuleList([
            nn.ModuleDict({
                'up': nn.Sequential(
                    nn.Conv2d(in_ch, out_ch * 4, 1),
                    nn.PixelShuffle(2)
                ),
                'process': SpectroTransBlock(out_ch)
            }) for in_ch, out_ch in decoder_channels
        ])
        
        # Final output layers
        self.output = nn.Conv2d(self.dims[0], inp_channels, 3, 1, 1)
    
    def forward(self, x):
        # Initial embedding
        feat = self.embed(x)
        skips = []
        
        # Encoding path
        for encoder in self.encoders:
            skips.append(feat)
            feat = encoder['down'](feat)
            feat, _ = encoder['process'](feat)
        
        # Middle processing
        for block in self.middle:
            feat, _ = block(feat)
        
        # Decoding path
        for decoder, skip in zip(self.decoders, reversed(skips)):
            feat = decoder['up'](feat)
            feat = feat + skip  # Skip connection
            feat, _ = decoder['process'](feat)
        
        # Output
        return self.output(feat) + x

if __name__ == '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis, flop_count_table, ActivationCountAnalysis
    import torch
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model initialization and analysis
    model = SpectroTemporalNet(dim=24).to(device)
    model.eval()
    
    # Parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f'\nTotal Parameters: {param_count:,}')
    
    # Create test input
    x = torch.randn(1, 3, 1024, 1024).to(device)
    
    with torch.no_grad():
        # Performance analysis
        print("\nAnalyzing model performance...")
        flops = FlopCountAnalysis(model, x)
        activations = ActivationCountAnalysis(model, x)
        
        try:
            print("\nModel Analysis:")
            print(flop_count_table(flops, activations))
        except Exception as e:
            print(f"Error during analysis: {e}")
            
        # Inference testing
        if device.type == 'cuda':
            model = model.half()
            x = x.half()
            
        # Warmup
        print("\nWarming up...")
        for _ in range(10):
            _ = model(x)
        torch.cuda.synchronize()
        
        # Benchmark
        print("\nRunning inference benchmark...")
        times = []
        for i in range(100):
            start_time = time.time()
            _ = model(x)
            torch.cuda.synchronize()
            times.append(time.time() - start_time)
            if (i + 1) % 20 == 0:
                print(f"Completed {i + 1}/100 iterations")
        
        # Results
        avg_time = sum(times) / len(times)
        print(f"\nBenchmark Results (1024x1024):")
        print(f"Average Inference Time: {avg_time:.4f}s")
        print(f"Frames Per Second: {1/avg_time:.2f}")
        
        if device.type == 'cuda':
            max_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"Peak GPU Memory Usage: {max_memory:.2f} MB")


# Average Inference Time: 0.0316s
