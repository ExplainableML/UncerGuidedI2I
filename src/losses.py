# from skimage.measure import compare_ssim as ssim
# from skimage.measure import compare_psnr as psnr
import torch
import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage.filters as fi
import matplotlib.pyplot as plt

# def compare_ssim(imgRef, imgT, K1=0.01, K2=0.03):
#     r = ssim(imgRef, imgT, data_range=imgT.max() - imgT.min(), multichannel=True, K1=K1, K2=K2)
#     return r

# def compare_psnr(imgRef, imgT):
#     r = psnr(imgRef, imgT, data_range=imgT.max() - imgT.min())
#     return r

# def compare_rrmse(imgRef, imgT):
#     numerator = (imgRef-imgT)**2
#     numerator = np.mean(numerator.flatten())
    
#     denominator = (imgRef)**2
#     denominator = np.mean(denominator.flatten())
    
#     r = numerator/denominator
#     r = np.sqrt(r)
#     return r

# def compare_qilv(I, I2, Ws=0.0, K1=0.01, K2=0.03):
#     C1 = K1**2
#     C2 = K2**2

#     kernsize=11
#     kernstd = 1.5
#     if Ws==0:
#         window = np.zeros((kernsize, kernsize))
#         window[kernsize//2, kernsize//2]=1
#         window = fi.gaussian_filter(window, kernstd)
#     window = window/np.sum(window)
    
#     chs = I.shape[2]
#     idxs = []
#     for ch in range(chs):
#         M1 = fi.convolve(I[:,:,ch], window)
#         M2 = fi.convolve(I2[:,:,ch], window)
#         Isq = I**2
#         I2sq = I2**2
#         V1 = fi.convolve(Isq[:,:,ch], window) - M1**2
#         V2 = fi.convolve(I2sq[:,:,ch], window) - M2**2

#         m1 = np.mean(V1)
#         m2 = np.mean(V2)
#         s1 = np.std(V1)
#         s2 = np.std(V2)
#         s12 = np.mean((V1-m1)*(V2-m2))

#         ind1 = (2*m1*m2+C1)/(m1**2+m2**2+C1)
#         ind2 = (2*s1*s2+C2)/(s1**2+s2**2+C2)
#         ind3 = (s12+C2/2)/(s1*s2+C2/2)
        
#         idxs.append(ind1*ind2*ind3)

#     return np.mean(idxs)

def bayeLq_loss(out_mean, out_log_var, target, q=2, k1=1, k2=1):
    var_eps = 1e-5
    out_var = var_eps + torch.exp(out_log_var)
    # out_log_var = torch.clamp(out_log_var, min=-3, max=3)
    # factor = torch.exp(-1*out_log_var) #no dropout grad_clipping b4 optim.step 
    factor = 1/out_var
    diffq = factor*torch.pow(torch.abs(out_mean-target), q)
#     diffq = torch.clamp(diffq, min=1e-5, max=1e3)
    
    loss1 = k1*torch.mean(diffq)
    loss2 = k2*torch.mean(torch.log(out_var))
    
    loss = 0.5*(loss1 + loss2)
    return loss

def bayeGen_loss(out_mean, out_1alpha, out_beta, target):
    alpha_eps, beta_eps = 1e-5, 1e-1
    out_1alpha += alpha_eps
    out_beta += beta_eps 
    factor = out_1alpha
    resi = torch.abs(out_mean - target)
#     resi = (torch.log((resi*factor).clamp(min=1e-4, max=5))*out_beta).clamp(min=-1e-4, max=5)
    resi = (resi*factor*out_beta).clamp(min=1e-6, max=50)
    log_1alpha = torch.log(out_1alpha)
    log_beta = torch.log(out_beta)
    lgamma_beta = torch.lgamma(torch.pow(out_beta, -1))
    
    if torch.sum(log_1alpha != log_1alpha) > 0:
        print('log_1alpha has nan')
        print(lgamma_beta.min(), lgamma_beta.max(), log_beta.min(), log_beta.max())
    if torch.sum(lgamma_beta != lgamma_beta) > 0:
        print('lgamma_beta has nan')
    if torch.sum(log_beta != log_beta) > 0:
        print('log_beta has nan')
    
    l = resi - log_1alpha + lgamma_beta - log_beta
    l = torch.mean(l)
    return l
    

def bayeLq_loss1(out_mean, out_var, target, q=2, k1=1, k2=1):
    '''
    out_var has sigmoid applied to it and is between 0 and 1
    '''
    eps = 1e-7
    out_log_var = torch.log(out_var + eps)
    factor = 1/(out_var + eps)
#     print('im dbg2: ', factor.min(), factor.max())
    diffq = factor*torch.pow(out_mean-target, q)
    loss1 = k1*torch.mean(diffq)
    loss2 = k2*torch.mean(out_log_var)
#     print('im dbg: ', loss1.item(), loss2.item())
    loss = 0.5*(loss1 + loss2)
    return loss

def bayeLq_loss_n_ch(out_mean, out_log_var, target, q=2, k1=1, k2=1, n_ch=3):
    '''
    assumes uncertainty values are single channel
    '''
    out_log_var_nch = out_log_var.repeat(1,n_ch,1,1)

    factor = torch.exp(-out_log_var_nch)
    diffq = factor*torch.pow(out_mean-target, q)
    loss1 = k1*torch.mean(diffq)
    loss2 = k2*torch.mean(out_log_var) #does it have to be nch times?
    loss = 0.5*(loss1 + loss2)
    return loss

def Sinogram_loss(A, out_y, target, q=2):
    '''
    A = n_rows x (128x88)
    expected image: 128 x 88
    So load the variable, transpose it.
    incoming variable: out_y, target: n_batch x 1 x 88 x 128

    z = out_y.view(-1,n_batch) : (128x88) x n_batch

    Az = n_row x 1
    '''
    n_batch = out_y.shape[0]
    #sino = torch.mm(A, out_y.view(-1,n_batch))
    #na = 120, nb = 128;
    #sino = sino.view(na,nb)
    resi = torch.abs(torch.mm(A, out_y.view(-1,n_batch)) - torch.mm(A, target.view(-1,n_batch)))
#     print('sino dbg1: ', resi.min(), resi.max())
    resi = torch.pow(resi, q)
    return torch.mean(resi)

def bayeLq_Sino_loss(A, out_mean, out_log_var, target, q=2, k1=1, k2=1):
    n_batch = out_mean.shape[0]
    var_eps = 3e-3
    out_var = var_eps + torch.exp(out_log_var)
    
    resi = torch.abs(torch.mm(A, out_mean.view(-1,n_batch)) - torch.mm(A, target.view(-1,n_batch)))
#     x1 = torch.mm(A, out_mean.view(-1,n_batch)).view(-1).data.cpu().numpy()
#     x2 = torch.mm(A, target.view(-1,n_batch)).view(-1).data.cpu().numpy()
#     plt.subplot(1,2,1)
#     plt.hist(x1)
#     plt.subplot(1,2,2)
#     plt.hist(x2)
#     plt.show()
    sino_var_eps = 2e-2
    A_out_log_var = torch.log(torch.mm(A, out_var.view(-1,n_batch)) + sino_var_eps)
#     print(A_out_log_var)
    x1 = A_out_log_var.view(-1).data.cpu().numpy()
#      plt.subplot(1,2,1)
#     plt.hist(x1)
#     plt.show()
    factor = torch.exp(-1*A_out_log_var)
    
    diffq = factor*torch.pow(resi, q)
    loss1 = k1*torch.mean(diffq)
    loss2 = k2*torch.mean(A_out_log_var)
    
    loss = 0.5*(loss1 + loss2)
    return loss

def bayeLq_Sino_loss1(A, out_mean, out_var, target, q=2, k1=1, k2=1):
    eps = 1e-7
    n_batch = out_mean.shape[0]
    #print(A.shape, out_mean.shape, out_log_var.shape, target.shape)
    resi = torch.abs(torch.mm(A, out_mean.view(-1,n_batch)) - torch.mm(A, target.view(-1,n_batch)))
    resi = torch.clamp(resi, min=0, max=1e2)
    
    out_log_var = torch.log(out_var+eps)
    A_out_log_var = torch.log(torch.mm(A, out_var.view(-1,n_batch)))
    A_out_log_var = torch.clamp(A_out_log_var, min=-3, max=3)
    
    factor = torch.exp(-1*A_out_log_var)
    
    diffq = factor*torch.pow(resi, q)
    loss1 = k1*torch.mean(diffq)
    loss2 = k2*torch.mean(A_out_log_var)
    
    loss = 0.5*(loss1 + loss2)
    return loss
    

def save_model(M, M_ckpt):
    torch.save(M.state_dict(), M_ckpt)
    print('model saved @ {}'.format(M_ckpt))

def show_G(G, x_lr, x_hr):
    G.eval()
    with torch.no_grad():
        plt.figure(figsize=(15,10))
        plt.subplot(1,5,1)
        plt.imshow(x_lr[0,0,:,:].data.cpu().numpy(), cmap='gray')
        plt.title('lr')

        mean_sr, log_var_sr = G(x_lr)
        var_sr = torch.exp(log_var_sr)
        plt.subplot(1,5,2)
        plt.imshow(mean_sr[0,0,:,:].data.cpu().numpy(), cmap='gray')
        plt.title('sr')
        
        plt.subplot(1,5,3)
        plt.imshow(log_var_sr[0,0,:,:].data.cpu().numpy(), cmap='jet')
        plt.title('log_var sr')
        plt.subplot(1,5,4)
        plt.imshow(var_sr[0,0,:,:].data.cpu().numpy(), cmap='jet')
        plt.title('var sr')

        plt.subplot(1,5,5)
        plt.imshow(x_hr[0,0,:,:].data.cpu().numpy(), cmap='gray')
        plt.title('hr')
        plt.show()

def Gen_loss(D_for_pred, pred, target, k1=1e-3):
    adv_loss = torch.mean(1 - D_for_pred)
    fid_loss = torch.nn.functional.mse_loss(pred, target)
    total_loss = fid_loss + k1*adv_loss
    return total_loss

def Gen_genUncer_loss(D_for_pred, pred, pred_1alpha, pred_beta, target, k1=1e-4):
    adv_loss = torch.mean(1 - D_for_pred)
    fid_loss = bayeGen_loss(pred, pred_1alpha, pred_beta, target)
    total_loss = fid_loss + k1*adv_loss
    return total_loss

def Dis_loss(D, SR_pred, HR_target):
    n_batch = SR_pred.shape[0]
    dtype = SR_pred.type()
    target_real = torch.rand(n_batch,1)*0.2 + 0.8
    target_fake = torch.rand(n_batch,1)*0.2
    target_real = target_real.type(dtype)
    target_fake = target_fake.type(dtype)
    
    adv_loss = torch.nn.functional.binary_cross_entropy(D(HR_target), target_real)
    adv_loss += torch.nn.functional.binary_cross_entropy(D(SR_pred), target_fake)
    return adv_loss