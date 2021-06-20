# UncerGuidedI2I
Uncertainty Guided Progressive GANs for Medical Image Translation

```
Image-to-image translation plays a vital role in tackling various medical imaging tasks such as attenuation correction, motion correction, undersampled reconstruction, and denoising. 
Generative adversarial networks have been shown to achieve the state-of-the-art in generating high fidelity images for these tasks.
However, the state-of-the-art GAN-based frameworks do not estimate the uncertainty in the predictions made by the network that is essential for making informed medical decisions 
and subsequent revision by medical experts and has recently been shown to improve the performance and interpretability of the model.
In this work, we propose an uncertainty-guided progressive learning scheme for image-to-image translation. 
By incorporating aleatoric uncertainty as attention maps for GANs trained in a progressive manner, we generate images of increasing fidelity progressively. 
We demonstrate the efficacy of our model on three challenging medical image translation tasks, 
including PET to CT translation, undersampled MRI reconstruction, and MRI motion artefact correction. 
Our model generalizes well in three different tasks and improves performance over state of the art under full-supervision and weak-supervision with limited data.
```
