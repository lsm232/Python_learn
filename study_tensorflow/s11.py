import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_absolute_error as compare_mae
import os
import argparse
from fix_seeds import *
from networks import *
import math
from loss import *
from load_img import *
import visdom
import time
import torch

#ssh -L 8096:127.0.0.1:8096 liaoshimin@10.2.5.38
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#参数设置----------------------------------------------------------------------------------------------
parser=argparse.ArgumentParser()
parser.add_argument('--n_epochs',type=int,default=200)
parser.add_argument('--batch_size',type=int,default=8)
parser.add_argument('--keep_lr_rate',type=int,default=200,help='训练过程中，保持初始学习率不变的epoch个数')
parser.add_argument('--lr_rate',type=float,default=0.00004)
parser.add_argument('--mask_ratio',type=float,default=0.6,help='训练过程中图像的掩码率')
parser.add_argument('--rec_ratio',type=float,default=0.6,help='训练过程中对图像的重建率，即对图像的rec_ratio*100/%进行重建')
parser.add_argument('--patch_size',type=int,default=8,help='训练过程中图像的掩码patch的尺寸')
parser.add_argument('--wp',type=int,default=12,help='计算掩码区域损失的权重')
parser.add_argument('--wr',type=int,default=1,help='计算未掩码区域损失的权重，即重构损失')
parser.add_argument('--wg',type=int,default=2,help='gan loss的权重')
parser.add_argument('--wa',type=int,default=10,help='自监督loss的权重')
parser.add_argument('--show_freq',type=int,default=20,help='可视化的频率，单位为迭代次数')
parser.add_argument('--cal_freq',type=int,default=5,help='计算模型指标的频率，单位为epoch')
parser.add_argument('--model_G_name',type=str,default='Unet_G_l0130-1_pretrain_gan-l',help='保存最佳生成器的名字')
parser.add_argument('--trainA_dir',type=str,default= r'/home/liaoshimin/self/data_all/stage1/train/low/',help='训练集低剂量CT图像路径')
parser.add_argument('--testA_dir',type=str,default= r'/home/liaoshimin/self/data_all/stage1/test/low/',help='测试集低剂量CT图像路径')
parser.add_argument('--port)',type=int,default=8096,help='可视化窗口')
opt=parser.parse_args()
#----------------------------------------------------------------------------------------------

#固定随机种子并加载生成器和判别器----------------------------------------------------------------
setup_seed(3)
G=Unet_G2(in_chans=2,out_chans=2).cuda()
D=Unet_D(in_chans=2,out_chans=2).cuda()
#----------------------------------------------------------------------------------------------

#优化器----------------------------------------------------------------------------------------
optimizer_G=torch.optim.AdamW(G.parameters(),lr=opt.lr_rate)
optimizer_D=torch.optim.AdamW(D.parameters(),lr=opt.lr_rate)
#----------------------------------------------------------------------------------------------

#学习率计划-------------------------------------------------------------------------------------
def rule(z):
    print("z==========:",z)
    return 1.0 - max(0, z - opt.keep_rate_epochs) / float(opt.n_epochs - opt.keep_rate_epochs + 1)
def rule2(z):
    print("z==========:", z)
    return ((1 + math.cos(z * math.pi / opt.n_epochs)) / 2) * (1 - 0.01) + 0.01
def rule3(z):
    return 1
scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,lr_lambda=lambda x: rule3(x))
scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D,lr_lambda=lambda x: rule3(x))
#-----------------------------------------------------------------------------------------------

#损失函数----------------------------------------------------------------------------------------
mse_loss=mse_fc.cuda()
mask_loss=mask_fc2
#------------------------------------------------------------------------------------------------

#加载训练集和测试集的数据--------------------------------------------------------------------------
train_data = ImageFolder_double(dir=opt.trainA_dir, transform=train_transform)
train_data_loader = data.DataLoader(
    dataset=train_data,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=8,
    drop_last=True,
)
test_data = ImageFolder_double(dir=opt.testA_dir, transform=test_transform)
test_data_loader = data.DataLoader(
    dataset=test_data,
    batch_size=1,
    shuffle=True,
    num_workers=2,
)
#-------------------------------------------------------------------------------------------------

#可视化--------------------------------------------------------------------------------------------
vis=visdom.Visdom(port=8096)
vis.line(Y=np.asarray([0.]),X=np.asarray([0.]),name='G',win='loss')
vis.line(Y=np.asarray([0.]),X=np.asarray([0.]),name='D',win='loss')
vis.images(torch.randn(size=(1,1,256,256)).numpy(),nrow=1,win='real')
vis.images(torch.randn(size=(1,1,256,256)).numpy(),nrow=1,win='fake')

vis.line(Y=np.asarray([0]), X=np.asarray([0]),
         win='test', name='G_test_mae')
vis.line(Y=np.asarray([0]), X=np.asarray([0]),
         win='test', name='G_test_psnr')
vis.line(Y=np.asarray([0]), X=np.asarray([0]),
         win='test', name='G_test_ssim')
#------------------------------------------------------------------------------------------------

#train-------------------------------------------------------------------------------------------
best_psnr=0.
best_mae=1000000.
best_ssim=0.
best_psnr_epoch=0
best_mae_epoch=0
best_ssim_epoch=0

for epoch in range(opt.n_epochs):
    start_time = time.time()
    G.train()
    D.train()
    old_lr = optimizer_G.param_groups[0]['lr']
    for step, (img1,img2,_) in enumerate(train_data_loader):
        img1=torch.clamp_max_(img1,1.0)
        img2=torch.clamp_max_(img2,1.0)
        img1 = img1.cuda()
        img2 = img2.cuda()
        masked_img1, mask1 = random_masking3(img1, opt.mask_ratio, opt.patch_size)
        masked_img2, mask2 = random_masking3(img2,opt.rec_ratio,opt.patch_size)

        reverse_mask2 = mask2.cpu().detach().numpy()
        reverse_mask2[reverse_mask2 == 0] = 100
        reverse_mask2[reverse_mask2 == 1] = 0
        reverse_mask2[reverse_mask2 == 100] = 1
        reverse_mask2 = torch.Tensor(reverse_mask2).cuda()

        # 训练生成器，此时判别器的参数没变
        real_imgs=torch.cat([img1,img2],dim=1)
        input_imgs=torch.cat([masked_img1,masked_img2],dim=1)
        fake_imgs=G(input_imgs)  #第一张图像用于计算预测损失，第二张图像用于计算重构损失
        set_requires_grad(D, False)
        optimizer_G.zero_grad()

        d_out = D(fake_imgs)  #判别器输出结果
        real_label = torch.ones_like(d_out)

        gan_loss = mse_loss(d_out, real_label)
        pre_loss = mask_loss(target=img1, pred=fake_imgs[0], mask=mask1, wp=1, wr=1)
        rec_loss = mask_loss(target=img2, pred=fake_imgs[1], mask=reverse_mask2, wp=1, wr=1)

        g_loss = opt.wg * gan_loss  + opt.wa * opt.wp * pre_loss + opt.wa * opt.wr * rec_loss
        g_loss.backward()
        optimizer_G.step()import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_absolute_error as compare_mae
import os
import argparse
from fix_seeds import *
from networks import *
import math
from loss import *
from load_img import *
import visdom
import time
import torch

#ssh -L 8096:127.0.0.1:8096 liaoshimin@10.2.5.38
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#参数设置----------------------------------------------------------------------------------------------
parser=argparse.ArgumentParser()
parser.add_argument('--n_epochs',type=int,default=200)
parser.add_argument('--batch_size',type=int,default=8)
parser.add_argument('--keep_lr_rate',type=int,default=200,help='训练过程中，保持初始学习率不变的epoch个数')
parser.add_argument('--lr_rate',type=float,default=0.00004)
parser.add_argument('--mask_ratio',type=float,default=0.6,help='训练过程中图像的掩码率')
parser.add_argument('--rec_ratio',type=float,default=0.6,help='训练过程中对图像的重建率，即对图像的rec_ratio*100/%进行重建')
parser.add_argument('--patch_size',type=int,default=8,help='训练过程中图像的掩码patch的尺寸')
parser.add_argument('--wp',type=int,default=12,help='计算掩码区域损失的权重')
parser.add_argument('--wr',type=int,default=1,help='计算未掩码区域损失的权重，即重构损失')
parser.add_argument('--wg',type=int,default=2,help='gan loss的权重')
parser.add_argument('--wa',type=int,default=10,help='自监督loss的权重')
parser.add_argument('--show_freq',type=int,default=20,help='可视化的频率，单位为迭代次数')
parser.add_argument('--cal_freq',type=int,default=5,help='计算模型指标的频率，单位为epoch')
parser.add_argument('--model_G_name',type=str,default='Unet_G_l0130-1_pretrain_gan-l',help='保存最佳生成器的名字')
parser.add_argument('--trainA_dir',type=str,default= r'/home/liaoshimin/self/data_all/stage1/train/low/',help='训练集低剂量CT图像路径')
parser.add_argument('--testA_dir',type=str,default= r'/home/liaoshimin/self/data_all/stage1/test/low/',help='测试集低剂量CT图像路径')
parser.add_argument('--port)',type=int,default=8096,help='可视化窗口')
opt=parser.parse_args()
#----------------------------------------------------------------------------------------------

#固定随机种子并加载生成器和判别器----------------------------------------------------------------
setup_seed(3)
G=Unet_G2(in_chans=2,out_chans=2).cuda()
D=Unet_D(in_chans=2,out_chans=2).cuda()
#----------------------------------------------------------------------------------------------

#优化器----------------------------------------------------------------------------------------
optimizer_G=torch.optim.AdamW(G.parameters(),lr=opt.lr_rate)
optimizer_D=torch.optim.AdamW(D.parameters(),lr=opt.lr_rate)
#----------------------------------------------------------------------------------------------

#学习率计划-------------------------------------------------------------------------------------
def rule(z):
    print("z==========:",z)
    return 1.0 - max(0, z - opt.keep_rate_epochs) / float(opt.n_epochs - opt.keep_rate_epochs + 1)
def rule2(z):
    print("z==========:", z)
    return ((1 + math.cos(z * math.pi / opt.n_epochs)) / 2) * (1 - 0.01) + 0.01
def rule3(z):
    return 1
scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,lr_lambda=lambda x: rule3(x))
scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D,lr_lambda=lambda x: rule3(x))
#-----------------------------------------------------------------------------------------------

#损失函数----------------------------------------------------------------------------------------
mse_loss=mse_fc.cuda()
mask_loss=mask_fc2
#------------------------------------------------------------------------------------------------

#加载训练集和测试集的数据--------------------------------------------------------------------------
train_data = ImageFolder_double(dir=opt.trainA_dir, transform=train_transform)
train_data_loader = data.DataLoader(
    dataset=train_data,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=8,
    drop_last=True,
)
test_data = ImageFolder_double(dir=opt.testA_dir, transform=test_transform)
test_data_loader = data.DataLoader(
    dataset=test_data,
    batch_size=1,
    shuffle=True,
    num_workers=2,
)
#-------------------------------------------------------------------------------------------------

#可视化--------------------------------------------------------------------------------------------
vis=visdom.Visdom(port=8096)
vis.line(Y=np.asarray([0.]),X=np.asarray([0.]),name='G',win='loss')
vis.line(Y=np.asarray([0.]),X=np.asarray([0.]),name='D',win='loss')
vis.images(torch.randn(size=(1,1,256,256)).numpy(),nrow=1,win='real')
vis.images(torch.randn(size=(1,1,256,256)).numpy(),nrow=1,win='fake')

vis.line(Y=np.asarray([0]), X=np.asarray([0]),
         win='test', name='G_test_mae')
vis.line(Y=np.asarray([0]), X=np.asarray([0]),
         win='test', name='G_test_psnr')
vis.line(Y=np.asarray([0]), X=np.asarray([0]),
         win='test', name='G_test_ssim')
#------------------------------------------------------------------------------------------------

#train-------------------------------------------------------------------------------------------
best_psnr=0.
best_mae=1000000.
best_ssim=0.
best_psnr_epoch=0
best_mae_epoch=0
best_ssim_epoch=0

for epoch in range(opt.n_epochs):
    start_time = time.time()
    G.train()
    D.train()
    old_lr = optimizer_G.param_groups[0]['lr']
    for step, (img1,img2,_) in enumerate(train_data_loader):
        img1=torch.clamp_max_(img1,1.0)
        img2=torch.clamp_max_(img2,1.0)
        img1 = img1.cuda()
        img2 = img2.cuda()
        masked_img1, mask1 = random_masking3(img1, opt.mask_ratio, opt.patch_size)
        masked_img2, mask2 = random_masking3(img2,opt.rec_ratio,opt.patch_size)

        reverse_mask2 = mask2.cpu().detach().numpy()
        reverse_mask2[reverse_mask2 == 0] = 100
        reverse_mask2[reverse_mask2 == 1] = 0
        reverse_mask2[reverse_mask2 == 100] = 1
        reverse_mask2 = torch.Tensor(reverse_mask2).cuda()

        # 训练生成器，此时判别器的参数没变
        real_imgs=torch.cat([img1,img2],dim=1)
        input_imgs=torch.cat([masked_img1,masked_img2],dim=1)
        fake_imgs=G(input_imgs)  #第一张图像用于计算预测损失，第二张图像用于计算重构损失
        set_requires_grad(D, False)
        optimizer_G.zero_grad()

        d_out = D(fake_imgs)  #判别器输出结果
        real_label = torch.ones_like(d_out)

        gan_loss = mse_loss(d_out, real_label)
        pre_loss = mask_loss(target=img1, pred=fake_imgs[0], mask=mask1, wp=1, wr=1)
        rec_loss = mask_loss(target=img2, pred=fake_imgs[1], mask=reverse_mask2, wp=1, wr=1)

        g_loss = opt.wg * gan_loss  + opt.wa * opt.wp * pre_loss + opt.wa * opt.wr * rec_loss
        g_loss.backward()
        optimizer_G.step()

        # 训练判别器
        set_requires_grad(D, True)
        optimizer_D.zero_grad()  # 梯度置0

        reals_out = D(real_imgs)  # 将真实图片输入判别器
        reals_label = torch.ones_like(reals_out)  # 1为真标签
        d_loss_real = mse_loss(reals_out, reals_label)  # 希望真实图片的输出接近1
        output = fake_imgs.detach()  # 生成器输出的假图片,这里的detach非常重要
        fakes_out = D(output)
        fakes_label = torch.zeros_like(fakes_out)  # 0为假标签
        d_loss_fake =mse_loss(fakes_out,fakes_label)   # 希望假图片的输出接近0
        d_loss = (d_loss_real + d_loss_fake) * opt.wg * 0.5
        d_loss.backward()
        optimizer_D.step()

        print('Epoch {:d}/{:d} G loss:{:.4f} D [real loss:{:.4f} fake loss:{:.4f}] auto loss:{:.4f}'.format(epoch + 1,
                                                                                                            opt.n_epochs,
                                                                                                            opt.gan_weight * g_loss.item(),
                                                                                                            gan_weight * d_loss_real.item(),
                                                                                                            gan_weight * d_loss_fake.item(),
                                                                                                            auto_weight * auto_loss.item()))




        # 训练判别器
        set_requires_grad(D, True)
        optimizer_D.zero_grad()  # 梯度置0

        reals_out = D(real_imgs)  # 将真实图片输入判别器
        reals_label = torch.ones_like(reals_out)  # 1为真标签
        d_loss_real = mse_loss(reals_out, reals_label)  # 希望真实图片的输出接近1
        output = fake_imgs.detach()  # 生成器输出的假图片,这里的detach非常重要
        fakes_out = D(output)
        fakes_label = torch.zeros_like(fakes_out)  # 0为假标签
        d_loss_fake =mse_loss(fakes_out,fakes_label)   # 希望假图片的输出接近0
        d_loss = (d_loss_real + d_loss_fake) * opt.wg * 0.5
        d_loss.backward()
        optimizer_D.step()

        print('Epoch {:d}/{:d} G loss:{:.4f} D [real loss:{:.4f} fake loss:{:.4f}] auto loss:{:.4f}'.format(epoch + 1,
                                                                                                            opt.n_epochs,
                                                                                                            opt.gan_weight * g_loss.item(),
                                                                                                            gan_weight * d_loss_real.item(),
                                                                                                            gan_weight * d_loss_fake.item(),
                                                                                                            auto_weight * auto_loss.item()))


