import torch
import torchio as tio
import matplotlib.pyplot as plt
import os, sys
import random
import nibabel as nib
import numpy as np
random.seed(0)

root_dir = './T1/'
out_dir = './MC_T1/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def get_mc_scan(src_scan, mc_tfm):
    '''
    src_scan should be normalized btw 0 to 1
    '''
    ret_scan = mc_tfm(src_scan)
    ret_scan = (ret_scan - ret_scan.min())/(ret_scan.max() - ret_scan.min())
    return ret_scan

mc_l0_tfm = tio.transforms.RandomMotion(degrees=10, translation=5, num_transforms=2, seed=0)
mc_l1_tfm = tio.transforms.RandomMotion(degrees=12, translation=6, num_transforms=2, seed=0)
mc_l2_tfm = tio.transforms.RandomMotion(degrees=14, translation=7, num_transforms=2, seed=0)
mc_l3_tfm = tio.transforms.RandomMotion(degrees=10, translation=2, num_transforms=3, seed=0)

all_nii_files = os.listdir(root_dir)

for l, mc_tfm in enumerate([mc_l3_tfm]):
    print('Generating for L{} ...'.format(l))
    save_dir = out_dir + 'L' + str(l) + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for f in all_nii_files:
        print('processing {} ...'.format(f))
        t1_image = tio.ScalarImage('{}{}'.format(root_dir, f))
        nib_t1 = nib.load('{}{}'.format(root_dir, f))

        # print(nib_t1.get_data().type, nib_t1.header.shape, nib_t1.get_data().shape)

        t1_in = t1_image.data
        t1_in = (t1_in - t1_in.min())/(t1_in.max() - t1_in.min())

        t1_l = get_mc_scan(t1_in, mc_tfm)
        
        t1_l_nib = nib.Nifti1Image(t1_l[0,:,:,:].numpy(), nib_t1.affine)
        nib.save(t1_l_nib, save_dir+f)