import numpy as np
import matplotlib.pyplot as plt
import os, sys
import nibabel as nib
import random
random.seed(0)

root_dir1 = './T1/'
root_dir2 = './MC_T1/L0/'
# root_dir3 = './MC_T1/L1/'
# root_dir4 = './MC_T1/L2/'
# root_dir5 = './MC_T1/L3/'

n_sub = 400
all_subjects = sorted(os.listdir(root_dir1))[:n_sub]

def create_out_dir_structure(out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path+'t1/')
        os.makedirs(out_path+'t1_l0/')
#         os.makedirs(out_path+'t1_l1/')
#         os.makedirs(out_path+'t1_l2/')
#         os.makedirs(out_path+'t1_l3/')
create_out_dir_structure('./mc_slices/')

def norm_0to1(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))

#dim0 -> coronal [90, 160] #blackout top 90 rows in t1
#dim1 -> axial [120, 190]
#dim2 -> saggital [30, 100] #blackout left 90 rows in t1

# # coronal
# for sub in all_subjects:
#     nmin = 90
#     nmax = 160
#     #extract from T2 mri
#     fnamet2 = root_dir+'T2/'+sub+'-T2.nii.gz'
#     fnamet1 = root_dir+'T1/'+sub+'-T1.nii.gz'
#     xt2 = nib.load(fnamet2).get_fdata()
#     xt2 = norm_0to1(xt2)
#     xt1 = nib.load(fnamet1).get_fdata()
#     xt1 = norm_0to1(xt1)
#     for i in range(nmin, nmax, 1):
#         xt1[i,0:75,:] = 0
#         np.save('./multimodal_slices/t2/{}_dim_{}_slice{}.npy'.format(sub,0,i), xt2[i,:,:])
#         np.save('./multimodal_slices/t1/{}_dim_{}_slice{}.npy'.format(sub,0,i), xt1[i,:,:])
#         # plt.subplot(1,2,1)
#         # plt.imshow(xt1[i,:,:], cmap='gray')
#         # plt.subplot(1,2,2)
#         # plt.imshow(xt2[i,:,:], cmap='gray')
#         # plt.show()  
#     print(sub, 'done!')

# axial
for sub in all_subjects:
    nmin = 120
    nmax = 190
    #extract from T2 mri
    fname1 = root_dir1+sub
    fname2 = root_dir2+sub
#     fname3 = root_dir3+sub
#     fname4 = root_dir4+sub
#     fname5 = root_dir5+sub

    x1 = nib.load(fname1).get_fdata()
    x1 = norm_0to1(x1)
    x2 = nib.load(fname2).get_fdata()
    x2 = norm_0to1(x2)
#     x3 = nib.load(fname3).get_fdata()
#     x3 = norm_0to1(x3)
#     x4 = nib.load(fname4).get_fdata()
#     x4 = norm_0to1(x4)
#     x5 = nib.load(fname5).get_fdata()
#     x5 = norm_0to1(x5)

    for i in range(nmin, nmax, 1):
        np.save('./mc_slices/t1/{}_dim_{}_slice{}.npy'.format(sub,1,i), x1[:,i,:])
        np.save('./mc_slices/t1_l0/{}_dim_{}_slice{}.npy'.format(sub,1,i), x2[:,i,:])
#         np.save('./mc_slices/t1_l1/{}_dim_{}_slice{}.npy'.format(sub,1,i), x3[:,i,:])
#         np.save('./mc_slices/t1_l2/{}_dim_{}_slice{}.npy'.format(sub,1,i), x4[:,i,:])
#         np.save('./mc_slices/t1_l3/{}_dim_{}_slice{}.npy'.format(sub,1,i), x5[:,i,:])
        # plt.subplot(1,2,1)
        # plt.imshow(xt1[:,i,:], cmap='gray')
        # plt.subplot(1,2,2)
        # plt.imshow(xt2[:,i,:], cmap='gray')
        # plt.show()  
    print(sub, 'done!')

# # sagital
# for sub in all_subjects:
#     nmin = 50
#     nmax = 100
#     #extract from T2 mri
#     fnamet2 = root_dir+'T2/'+sub+'-T2.nii.gz'
#     fnamet1 = root_dir+'T1/'+sub+'-T1.nii.gz'
#     xt2 = nib.load(fnamet2).get_fdata()
#     xt2 = norm_0to1(xt2)
#     xt1 = nib.load(fnamet1).get_fdata()
#     xt1 = norm_0to1(xt1)
#     for i in range(nmin, nmax, 1):
#         xt1[:,0:65,i] = 0
#         np.save('./multimodal_slices/t2/{}_dim_{}_slice{}.npy'.format(sub,2,i), xt2[:,:,i])
#         np.save('./multimodal_slices/t1/{}_dim_{}_slice{}.npy'.format(sub,2,i), xt1[:,:,i])
#         # plt.subplot(1,2,1)
#         # plt.imshow(xt1[:,:,i], cmap='gray')
#         # plt.subplot(1,2,2)
#         # plt.imshow(xt2[:,:,i], cmap='gray')
#         # plt.show()  
#     print(sub, 'done!')