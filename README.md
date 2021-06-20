# UncerGuidedI2I
Uncertainty Guided Progressive GANs for Medical Image Translation

![](./UncerGuidedI2I_Model.gif)

This repository provides the code for the MICCAI-2021 paper "Uncertainty-guided Progressive GANs for Medical Image Translation". 
We take inspiration from the progressive learning scheme demonstrated at MedGAN and Progressive GANs, and augment the learning with intermediate uncertainty maps, that are used as attention map to focus the image translation in poorly constructed (highly uncertain) regions.

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
### Preparing Datasets


