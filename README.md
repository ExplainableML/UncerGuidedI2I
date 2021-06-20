# UncerGuidedI2I
Uncertainty Guided Progressive GANs for Medical Image Translation

![](./UncerGuidedI2I_Model.gif)

This repository provides the code for the MICCAI-2021 paper "Uncertainty-guided Progressive GANs for Medical Image Translation". 
We take inspiration from the progressive learning scheme demonstrated at MedGAN and Progressive GANs, and augment the learning with the estimation of intermediate uncertainty maps, that are used as attention map to focus the image translation in poorly constructed (highly uncertain) regions, progressively improving the images over multiple phases.

![](./UncerGuidedI2I_res.gif)

The structure of the repository is as follows:
```
root
 |-ckpt/ (will save all the checkpoints)
 |-data/ (save your data and related script)
 |-src/ (contains all the source code)
    |-ds.py 
    |-networks.py
    |-utils.py
    |-losses.py
```

## How to use
### Requirements
```
pytorch > 1.6.0
torchio
scikit-image
scikit-learn
```

### Preparing Datasets
The experiments of the paper used T1 MRI scans from the IXI dataset and a propietary PET/CT dataset.

`data/IXI/` has jupyter notebooks to prepare the data for motion correction as well as undersampled MRI reconstruction.
For custom datasets, use the above notebooks as example to prepare the dataset and place them under `data/`. The dataset class in `src/ds.py` loads the paired set of images (corrupted and the non-corrupted version).

### Training
`src/networks.py` provides the generator and discriminator architectures.

`src/utils.py` provides two training APIs `train_i2i_UNet3headGAN` and `train_i2i_Cas_UNet3headGAN`. The first API is to be used to train the primary GAN, whereas the second API is to be used to train the subsequent GANs. 

An example command to use the first API is:
```python
netG_A = CasUNet_3head(1,1)
netD_A = NLayerDiscriminator(1, n_layers=4)
netG_A, netD_A = train_i2i_UNet3headGAN(
    netG_A, netD_A,
    train_loader, test_loader,
    dtype=torch.cuda.FloatTensor,
    device='cuda',
    num_epochs=50,
    init_lr=1e-5,
    ckpt_path='../ckpt/i2i_UNet3headGAN',
)
```

An example command to use the first API is:
```python
netG_A1 = CasUNet_3head(1,1)
netG_A1.load_state_dict(torch.load('../ckpt/uncorr2CT_UNet3headGAN_v1_eph78_G_A.pth'))
netG_A2 = UNet_3head(4,1)
netG_A2.load_state_dict(torch.load('../ckpt/uncorr2CT_Cas_UNet3headGAN_v1_eph149_G_A.pth'))
netG_A3 = UNet_3head(4,1)

netD_A = NLayerDiscriminator(1, n_layers=4)
list_netG_A, list_netD_A = train_uncorr2CT_Cas_UNet3headGAN(
    [netG_A1, netG_A2, netG_A3], [netD_A],
    train_loader, test_loader,
    dtype=torch.cuda.FloatTensor,
    device='cuda',
    num_epochs=50,
    init_lr=1e-5,
    ckpt_path='../ckpt/uncorr2CT_Cas_UNet3headGAN_v1_block3',
    noise_sigma=0.0
)
```