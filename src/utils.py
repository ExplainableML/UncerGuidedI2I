import torch
import numpy as np
import scipy as sp
import skimage
from scipy.special import gamma, factorial
import matplotlib.gridspec as gridspec
from scipy.stats import gennorm
# import seaborn as sns
# sns.set_style('darkgrid')
import os, sys
import PIL
from PIL import Image
from matplotlib import pyplot as plt
from losses import *
from networks import *
from ds import *
import random
random.seed(0)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim
from torchvision import transforms, utils as tv_utils

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor


def train_i2i_UNet3headGAN(
    netG_A,
    netD_A,
    train_loader, test_loader,
    dtype=torch.cuda.FloatTensor,
    device='cuda',
    num_epochs=50,
    init_lr=1e-4,
    ckpt_path='../ckpt/i2i_UNet3headGAN',
):
    netG_A.to(device)
    netG_A.type(dtype)
    ####
    netD_A.to(device)
    netD_A.type(dtype)
    
    ####
    optimizerG = torch.optim.Adam(list(netG_A.parameters()), lr=init_lr)
    optimizerD = torch.optim.Adam(list(netD_A.parameters()), lr=init_lr)
    optimG_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, num_epochs)
    optimD_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, num_epochs)
    ####
    list_epochs = [50, 50, 150]
    list_lambda1 = [1, 0.5, 0.1]
    list_lambda2 = [0.0001, 0.001, 0.01]
    for num_epochs, lam1, lam2 in zip(list_epochs, list_lambda1, list_lambda2):
        for eph in range(num_epochs):
            netG_A.train()
            netD_A.train()
            avg_rec_loss = 0
            avg_tot_loss = 0
            print(len(train_loader))
            for i, batch in enumerate(train_loader):
                if i>1000:
                    break
                xA, xB = batch[0].to(device).type(dtype), batch[1].to(device).type(dtype)
                #calc all the required outputs
                rec_B, rec_alpha_B, rec_beta_B = netG_A(xA)

                #first gen
                netD_A.eval()
                total_loss = lam1*F.l1_loss(rec_B, xB) + lam2*bayeGen_loss(rec_B, rec_alpha_B, rec_beta_B, xB)
                t0 = netD_A(rec_B)
                t1 = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                e5 = 0.001*F.mse_loss(t1, torch.ones(t1.size()).to(device).type(dtype))
                total_loss += e5
                optimizerG.zero_grad()
                total_loss.backward()
                optimizerG.step()

                #then discriminator
                netD_A.train()
                t0 = netD_A(xB)
                pred_real_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                loss_D_A_real = 1*F.mse_loss(
                    pred_real_A, torch.ones(pred_real_A.size()).to(device).type(dtype)
                )
                t0 = netD_A(rec_B.detach())
                pred_fake_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                loss_D_A_pred = 1*F.mse_loss(
                    pred_fake_A, torch.zeros(pred_fake_A.size()).to(device).type(dtype)
                )
                loss_D_A = (loss_D_A_real + loss_D_A_pred)*0.5

                loss_D = loss_D_A
                optimizerD.zero_grad()
                loss_D.backward()
                optimizerD.step()

                avg_tot_loss += total_loss.item()

            avg_tot_loss /= len(train_loader)
            print(
                'epoch: [{}/{}] | avg_tot_loss: {}'.format(
                    eph, num_epochs, avg_tot_loss
                )
            )
            torch.save(netG_A.state_dict(), ckpt_path+'_eph{}_G_A.pth'.format(eph))
            torch.save(netD_A.state_dict(), ckpt_path+'_eph{}_D_A.pth'.format(eph))
    return netG_A, netD_A


def train_i2i_Cas_UNet3headGAN(
    list_netG_A,
    list_netD_A,
    train_loader, test_loader,
    dtype=torch.cuda.FloatTensor,
    device='cuda',
    num_epochs=50,
    init_lr=1e-4,
    ckpt_path='../ckpt/i2i_UNet3headGAN',
):
    for nid, m1 in enumerate(list_netG_A):
        m1.to(device)
        m1.type(dtype)
        list_netG_A[nid] = m1
        
    for nid, m2 in enumerate(list_netD_A):
        m2.to(device)
        m2.type(dtype)
        list_netD_A[nid] = m2
    ####
    optimizerG = torch.optim.Adam(list(list_netG_A[-1].parameters()), lr=init_lr)
    optimizerD = torch.optim.Adam(list(list_netD_A[-1].parameters()), lr=init_lr)
    optimG_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, num_epochs)
    optimD_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, num_epochs)
    ####
    list_epochs = [50, 50, 150]
    list_lambda1 = [1, 0.5, 0.1]
    list_lambda2 = [0.0001, 0.001, 0.01]
    netG_A, netD_A = list_netG_A[-1], list_netD_A[-1]
    ####
    for num_epochs, lam1, lam2 in zip(list_epochs, list_lambda1, list_lambda2):
        for eph in range(num_epochs):
            netG_A.train()
            netD_A.train()
            avg_rec_loss = 0
            avg_tot_loss = 0
            print(len(train_loader))
            for i, batch in enumerate(train_loader):
                if i>1000:
                    break
                xA, xB = batch[0].to(device).type(dtype), batch[1].to(device).type(dtype)
                #calc all the required outputs
                
                for nid, netG in enumerate(list_netG_A):
                    if nid == 0:
                        rec_B, rec_alpha_B, rec_beta_B = netG(xA)
                    else:
                        xch = torch.cat([rec_B, rec_alpha_B, rec_beta_B, xA], dim=1)
                        rec_B, rec_alpha_B, rec_beta_B = netG(xch)

                #first gen
                netD_A.eval()
                total_loss = lam1*F.l1_loss(rec_B, xB) + lam2*bayeGen_loss(rec_B, rec_alpha_B, rec_beta_B, xB)
                t0 = netD_A(rec_B)
                t1 = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                e5 = 0.001*F.mse_loss(t1, torch.ones(t1.size()).to(device).type(dtype))
                total_loss += e5
                optimizerG.zero_grad()
                total_loss.backward()
                optimizerG.step()

                #then discriminator
                netD_A.train()
                t0 = netD_A(xB)
                pred_real_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                loss_D_A_real = 1*F.mse_loss(
                    pred_real_A, torch.ones(pred_real_A.size()).to(device).type(dtype)
                )
                t0 = netD_A(rec_B.detach())
                pred_fake_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                loss_D_A_pred = 1*F.mse_loss(
                    pred_fake_A, torch.zeros(pred_fake_A.size()).to(device).type(dtype)
                )
                loss_D_A = (loss_D_A_real + loss_D_A_pred)*0.5

                loss_D = loss_D_A
                optimizerD.zero_grad()
                loss_D.backward()
                optimizerD.step()

                avg_tot_loss += total_loss.item()

                if i%500 == 0:
                    print(eph, i)
                    test_uncorr2CT_Cas_UNet3headGAN_n_show(
                        list_netG_A,
                        test_loader,
                        device,
                        dtype,
                        nrow=1,
                        n_show = 1
                    )
            avg_tot_loss /= len(train_loader)
            print(
                'epoch: [{}/{}] | avg_tot_loss: {}'.format(
                    eph, num_epochs, avg_tot_loss
                )
            )
            torch.save(netG_A.state_dict(), ckpt_path+'_eph{}_G_A.pth'.format(eph))
            torch.save(netD_A.state_dict(), ckpt_path+'_eph{}_D_A.pth'.format(eph))
    return list_netG_A, list_netD_A

# # init net and train
# netG_A = CasUNet_3head(1,1)
# netD_A = NLayerDiscriminator(1, n_layers=4)
# netG_A, netD_A = train_i2i_UNet3headGAN(
#     netG_A, netD_A,
#     train_loader, test_loader,
#     dtype=torch.cuda.FloatTensor,
#     device='cuda',
#     num_epochs=50,
#     init_lr=1e-5,
#     ckpt_path='../ckpt/i2i_UNet3headGAN',
# )

# # init net and train
# netG_A1 = CasUNet_3head(1,1)
# netG_A1.load_state_dict(torch.load('../ckpt/uncorr2CT_UNet3headGAN_v1_eph78_G_A.pth'))
# netG_A2 = UNet_3head(4,1)
# netG_A2.load_state_dict(torch.load('../ckpt/uncorr2CT_Cas_UNet3headGAN_v1_eph149_G_A.pth'))
# netG_A3 = UNet_3head(4,1)

# netD_A = NLayerDiscriminator(1, n_layers=4)
# list_netG_A, list_netD_A = train_uncorr2CT_Cas_UNet3headGAN(
#     [netG_A1, netG_A2, netG_A3], [netD_A],
#     train_loader, test_loader,
#     dtype=torch.cuda.FloatTensor,
#     device='cuda',
#     num_epochs=50,
#     init_lr=1e-5,
#     ckpt_path='../ckpt/uncorr2CT_Cas_UNet3headGAN_v1_block3',
#     noise_sigma=0.0
# )