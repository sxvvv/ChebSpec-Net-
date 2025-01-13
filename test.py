import os
import shutil
import torch
import argparse
from tqdm import tqdm
from utils.val_utils import AverageMeter, compute_psnr_ssim
from net.model import SpectroTemporalNet  
import pytorch_lightning as pl
import torch.nn as nn
from utils.dataset_utils import TestDataset, DenoiseTestDataset
import numpy as np
from utils.image_io import save_image_tensor
from utils.loss_utils import*

class UHDModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = SpectroTemporalNet()
        self.loss_f1 = L1Loss(loss_weight=1.0, reduction='mean')
        self.difficult_samples = []
    
    def forward(self,x):
        restored = self.net(x)
        return restored
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        (degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_l1(restored,clean_patch) 
        self.log("train_loss", loss)
        
        return loss

if __name__ == '__main__':
    # 定义命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=3)
    parser.add_argument('--valid_data_dir', type=str, default="/data/SDA/suxin/Multitask/UHD-Snow/test/") # epoch=751 PSNR: 37.00, SSIM: 0.9877
    
    parser.add_argument('--output_path', type=str, default="output/")
    parser.add_argument('--ckpt_path', type=str, default="/data/SDA/suxin/FGSSM/UHDSnow/last.ckpt")
    args = parser.parse_args()

    # 设置随机种子和CUDA设备
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(args.cuda)

    # 加载模型和权重
    ckpt_path = args.ckpt_path
    print("CKPT name : {}".format(ckpt_path))

    net = UHDModel().load_from_checkpoint(ckpt_path).cuda()
    net.eval()
 
    # 创建数据集和数据加载器
    data_path = args.valid_data_dir
    dataset_name = data_path.split('/')[-2]
    print(f'Test: {dataset_name}')
    
    data_set = TestDataset(args)

    output_path = args.output_path + dataset_name + '/'
    testloader = torch.utils.data.DataLoader(data_set, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    
    # 初始化PSNR和SSIM的列表
    psnr_list = []
    ssim_list = []
    names_list = []
    psnr = AverageMeter()
    ssim = AverageMeter()
    # 计算每张图像的PSNR和SSIM
    with torch.no_grad():
        for (degraded_name, degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            print(degraded_name)
            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            print(temp_ssim)
            psnr_list.append(temp_psnr)
            ssim_list.append(temp_ssim)
            names_list.append(degraded_name[0])
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            print(temp_psnr)
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg)) 
