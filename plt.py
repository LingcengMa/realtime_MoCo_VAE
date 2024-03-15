import os
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
from vae import VariationalAutoencoder
import torchvision
import numpy as np
import scipy
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

#data
data_file = os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'realtime_GT_imgs.mat')
mat = scipy.io.loadmat(data_file)
data = mat['Phi_rt_init'] 
data_real = data.real
data_imag = data.imag
data_con = np.concatenate((data_real, data_imag))
data_con = data_con.transpose()
data = torch.from_numpy(data_con).float()



seg_file = os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'TI.mat')
segidx = scipy.io.loadmat(seg_file)
segidx = segidx['Segidx']
# segidx_mu = (segidx - segidx.mean())/segidx.std()

binning_file = os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'binning_GT.mat')
binning_file = scipy.io.loadmat(binning_file)
hidx = binning_file['Hidx'] # Cardiac
ridx = binning_file['Ridx'] # Respiratory
ridx_mu = (ridx - ridx.mean())/ridx.std()
hidx_mu = (hidx - hidx.mean())/hidx.std()
hidx_sin = np.sin(hidx*2*np.pi/20)
hidx_cos = np.cos(hidx*2*np.pi/20)

array_1 = np.ones(360)
array_2 = 2 * array_1
array_3 = 3 * array_1
array_4 = 4 * array_1
array_5 = 5 * array_1
array = np.concatenate((array_1, array_2, array_3),axis=0)
array = np.concatenate((array, array_4, array_5),axis=0)
array = np.tile(array, 7)

Ti = segidx*2*3.56e-3
T1 = 1.2
Tr = 3.56e-3
alpha = 5*np.pi/180
T1_star = 1/(1/T1 - np.log(np.cos(alpha))/Tr)
u1 = 1 - 2 * np.exp(-Ti/T1_star)

# create ramp matrix
array_1 = np.ones(360)
array_2 = 2 * array_1
array_3 = 3 * array_1
array_4 = 4 * array_1
array_5 = 5 * array_1
array = np.concatenate((array_1, array_2, array_3),axis=0)
array = np.concatenate((array, array_4, array_5),axis=0)
wall_clock = np.tile(array, 7)

tau = (wall_clock - 1) * 15e-3
T2 = 50e-3
u2 = np.exp(-tau/T2)

# plt.plot(u1)
# plt.xlim(0,1800)
# plt.show()

wall_clock_mu = (wall_clock-wall_clock.mean())/wall_clock.std()
segidx_mu = (segidx-segidx.mean())/segidx.std()

# filtered_ridx_mu = (filtered_ridx - filtered_ridx.mean())/filtered_ridx.std()

u1_normalized = (u1 - u1.mean())/u1.std()
u2_normalized = (u2 - u2.mean())/u2.std()

# # time domain plot
# plt.figure(1)
# plt.subplot(2,2,1)
# plt.plot(segidx)
# plt.xlim(0,1800)
# plt.xlabel('Time')
# plt.title('Linear T1 Representation')
# plt.grid(True)

# plt.subplot(2,2,2)
# plt.plot(u1)
# plt.xlim(0,1800)
# plt.xlabel('Time')
# plt.title('Nonlinear T1 Representation')
# plt.grid(True)

# plt.subplot(2,2,3)
# plt.plot(wall_clock)
# plt.xlim(0,1800)
# plt.xlabel('Time')
# plt.title('Linear T2 Representation')
# plt.grid(True)

# plt.subplot(2,2,4)
# plt.plot(u2)
# plt.xlim(0,1800)
# plt.xlabel('Time')
# plt.title('Noninear T2 Representation')
# plt.grid(True)

# plt.tight_layout()
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/input_signal_t1t2.png')

# # time domain plot
# plt.figure(2)
# plt.subplot(2,2,1)
# plt.plot(hidx)
# plt.xlim(0,2000)
# plt.xlabel('Time')
# plt.title('Phase Cardiac Representation')
# plt.grid(True)

# plt.subplot(2,2,2)
# plt.plot(ridx)
# plt.xlim(0,2000)
# plt.xlabel('Time')
# plt.title('Phase Respiratory Representation')
# plt.grid(True)

# plt.subplot(2,2,3)
# plt.plot(hidx_cos)
# plt.xlim(0,2000)
# plt.xlabel('Time')
# plt.title('Cosine Cardiac Representation')
# plt.grid(True)

# plt.subplot(2,2,4)
# plt.plot(hidx_sin)
# plt.xlim(0,2000)
# plt.xlabel('Time')
# plt.title('Sine Cardiac Representation')
# plt.grid(True)

# plt.tight_layout()
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/input_signal_motion.png')


train_losses = np.load('train_losses_6.npy')
plt.plot(train_losses)
plt.show()




# u1_normalized = torch.from_numpy(u1_normalized)
# u2_normalized = torch.from_numpy(u2_normalized)
# wall_clock_mu = torch.from_numpy(wall_clock_mu)
# segidx_mu  = torch.from_numpy(segidx_mu)
# hidx_mu = torch.from_numpy(hidx_mu)
# hidx_sin = torch.from_numpy(hidx_sin)
# hidx_cos = torch.from_numpy(hidx_cos)
# ridx_mu = torch.from_numpy(ridx_mu)
# array_mu = (array - array.mean())/array.std()

# noise = np.random.uniform(-0.5, 0.5, size=(12600,1))
# noisy_hidx = hidx + noise
# noisy_ridx = ridx + noise


# latent_4 = np.load('latent_LT1_LT2_Sin.npy')


# # time domain plot
# plt.figure(1)
# plt.subplot(2,2,1)
# plt.plot(latent_4[:,0])
# plt.xlim(0,1800)
# plt.ylabel('Latent Vector 1')
# plt.xlabel('time')

# plt.subplot(2,2,2)
# plt.plot(latent_4[:,1])
# plt.xlim(0,1800)
# plt.ylabel('Latent Vector 2')
# plt.xlabel('time')

# plt.subplot(2,2,3)
# plt.plot(latent_4[:,2])
# plt.xlim(0,1800)
# plt.ylabel('Latent Vector 3')
# plt.xlabel('time')

# plt.subplot(2,2,4)
# plt.plot(latent_4[:,3])
# plt.xlim(0,1800)
# plt.ylabel('Latent Vector 4')
# plt.xlabel('time')
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/LT1_LT2_Sin.png')


# # scatter plot
# plt.figure(2)
# plt.scatter(ridx, latent_4[:,3], marker='o', s=3, color='blue', alpha=0.5)
# plt.ylabel('Learned')
# plt.xlabel('Ground truth')
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/scatter_LT1_LT2_Sin.png')


# latent_4 = np.load('latent_LT1_NT2_Phase.npy')

# # time domain plot
# plt.figure(3)
# plt.subplot(2,2,1)
# plt.plot(latent_4[:,0])
# plt.xlim(0,1800)
# plt.ylabel('Latent Vector 1')
# plt.xlabel('time')

# plt.subplot(2,2,2)
# plt.plot(latent_4[:,1])
# plt.xlim(0,1800)
# plt.ylabel('Latent Vector 2')
# plt.xlabel('time')

# plt.subplot(2,2,3)
# plt.plot(latent_4[:,2])
# plt.xlim(0,1800)
# plt.ylabel('Latent Vector 3')
# plt.xlabel('time')

# plt.subplot(2,2,4)
# plt.plot(latent_4[:,3])
# plt.xlim(0,1800)
# plt.ylabel('Latent Vector 4')
# plt.xlabel('time')
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/LT1_NT2_Phase.png')


# # scatter plot
# plt.figure(4)
# plt.scatter(ridx, latent_4[:,3], marker='o', s=3, color='blue', alpha=0.5)
# plt.ylabel('Learned')
# plt.xlabel('Ground truth')
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/scatter_LT1_NT2_Phase.png')

# latent_4 = np.load('latent_NT1_LT2_CosSin.npy')

# # time domain plot
# plt.figure(5)
# plt.subplot(3,1,1)
# plt.plot(latent_4[:,2])
# plt.xlim(0,1800)
# plt.ylabel('Latent Vector 3')
# plt.xlabel('time')

# plt.subplot(3,1,2)
# plt.plot(latent_4[:,3])
# plt.xlim(0,1800)
# plt.ylabel('Latent Vector 5')
# plt.xlabel('time')

# plt.subplot(3,1,3)
# plt.plot(latent_4[:,4])
# plt.xlim(0,1800)
# plt.ylabel('Latent Vector 5')
# plt.xlabel('time')

# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/NT1_LT2_CosSine.png')


# # scatter plot
# plt.figure(6)
# plt.scatter(ridx, latent_4[:,4], marker='o', s=3, color='blue', alpha=0.5)
# plt.ylabel('Learned')
# plt.xlabel('Ground truth')
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/scatter_NT1_LT2_CosSine.png')






# plt.figure(2)
# plt.scatter(array, latent_4[:,3], marker='o', s=3, color='blue', alpha=0.5)
# plt.ylabel('Latent Vector 4')
# plt.xlabel('segidx')
# # plt.ylim(-2,4)
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/latent_4d3_vs_segidx_hc_f.png')

# plt.figure(3)
# fft_latent_3 = np.fft.fft(latent_4[:, 2])
# frequency = np.fft.fftfreq(segidx.size,d=0.007)
# plt.plot(frequency, np.abs(fft_latent_3))
# plt.title('FFT of Latent Vector 3')
# plt.xlim(-2,2)
# plt.xlabel('Frequency')
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/latent_4d3_fft_hc_f.png')

# plt.figure()
# fft_latent_4 = np.fft.fft(latent_4[:, 3])
# frequency = np.fft.fftfreq(segidx.size,d=0.007)
# plt.plot(frequency, np.abs(fft_latent_4), label = 'learned')
# plt.plot(frequency, np.abs(np.fft.fft(ridx_mu)), label = 'ground truth')
# plt.title('FFT of Latent Vector 4')
# plt.xlim(-2,2)
# plt.xlabel('Frequency')
# plt.legend()
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/test.png')

# # plt.figure(5)
# # plt.plot(latent_4[:,0])
# # plt.xlim(0,1800)
# # plt.show()





# plt.scatter(ridx_mu, latent_2[:,0], marker='o', s=3, color='blue', alpha=0.5)
# plt.ylabel('Latent Vector 1')
# plt.xlabel('ridx')
# # plt.ylim(-2,4)
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/1.png')

# latent_3 = np.load('latent_3dim.npy')

# plt.scatter(ridx_mu, latent_2[:,0], marker='o', s=3, color='blue', alpha=0.5)
# plt.ylabel('Latent Vector 1')
# plt.xlabel('ridx')
# # plt.ylim(-2,4)
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/1.png')

# fft_latent_2 = np.fft.fft(latent_2[:, 1])
# frequency = np.fft.fftfreq(segidx.size,d=0.007)
# plt.plot(frequency, np.abs(fft_latent_2))
# plt.title('FFT of Latent Vector 2')
# plt.xlim(-2,2)
# plt.xlabel('Frequency')
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/latent_2_k_1_5_fft1.png')









# plt.scatter(segidx, latent_1, marker='o', s=3, color='blue', alpha=0.5)
# plt.ylabel('Latent Vector 1')
# plt.xlabel('Segidx')
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/latent_1_ref_vs_Segidx_(1).png')


# plt.scatter(noisy_hidx, latent_1, marker='o', s=3, color='blue', alpha=0.5)
# plt.ylabel('Latent Vector 1')
# plt.xlabel('Hidx')
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/latent_1_1dim_Hidx_noise.png')

# plt.scatter(noisy_ridx, latent_1, marker='o', s=3, color='blue', alpha=0.5)
# plt.ylabel('Latent Vector 1')
# plt.xlabel('Ridx')
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/latent_1_1dim_Ridx_noise.png')

# plt.hist2d(hidx.transpose(), latent_1.transpose(), bins=20, cmap='Blues')
# plt.ylabel('Latent Vector 1')
# plt.xlabel('Hidx')
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/latent_1_1dim_Hidx_hist.png')

# plt.hist2d(ridx.transpose(), latent_1.transpose(), bins=20, cmap='Blues')
# plt.ylabel('Latent Vector 1')
# plt.xlabel('Ridx')
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/latent_1_1dim_Ridx_hist.png')

 
# plt.scatter(ridx, latent_3[:,0], marker='o', s=3, color='blue', alpha=0.5)
# plt.ylabel('Latent Vector 1')
# plt.xlabel('Ridx')

# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/latent_1_3dim_Ridx.png')

# fft_dim2 = np.fft.fft(latent_3[:, 1])
# fft_dim3 = np.fft.fft(latent_3[:, 2])
# frequency = np.fft.fftfreq(segidx.size,d=0.007)

# # Create a plot for the FFT of each dimension
# plt.subplot(1, 2, 1)
# plt.plot(frequency, np.abs(fft_dim2))
# plt.title('FFT of Latent Vector 2')
# plt.xlim(-2,2)
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')

# plt.subplot(1, 2, 2)
# plt.plot(frequency, np.abs(fft_dim3))
# plt.title('FFT of Latent Vector 3')
# plt.xlim(-2,2)
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')

# plt.tight_layout()
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/latent_23_3dim_lim2.png')

# plot
# latent_2 = vae.encoder(data)
# latent_2 = latent_2.to('cpu').detach().numpy()

# fft_dim2 = np.fft.fft(latent_2[:, 1])
# frequency = np.fft.fftfreq(segidx.size)

# # Create a plot for the FFT of each dimension

# plt.plot(frequency, np.abs(fft_dim2))
# plt.title('FFT of Latent Vector 2')
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')
# plt.show()

# 
# plt.subplot(3,1,1)
# plt.scatter(segidx, latent_3[:,0], marker='o', s=3, color='blue', alpha=0.5)
# plt.ylabel('Latent Vector 1')
# plt.xlabel('Segidx')
# # plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/latent_1_3dim_Segidx_1.png')

# plt.subplot(1,2,1)
# plt.scatter(hidx, latent_3[:,0], marker='o', s=3, color='blue', alpha=0.5)
# plt.ylabel('Latent Vector 1')
# plt.xlabel('Hidx')
# # plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/latent_1_3dim_Hidx_1.png')

# plt.subplot(1,2,2)
# plt.scatter(ridx, latent_3[:,0], marker='o', s=3, color='blue', alpha=0.5)
# plt.ylabel('Latent Vector 1')
# plt.xlabel('Ridx')
# # plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/latent_1_3dim_Ridx_1.png')

# plt.tight_layout()
# plt.show()

# fft_dim2 = np.fft.fft(latent_3[:, 1])
# fft_dim3 = np.fft.fft(latent_3[:, 2])
# frequency = np.fft.fftfreq(segidx.size,d=0.007)

# Create a plot for the FFT of each dimension
# plt.subplot(1, 2, 1)
# plt.plot(frequency, np.abs(fft_dim2))
# plt.title('FFT of Latent Vector 2')
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')

# plt.subplot(1, 2, 2)
# plt.plot(frequency, np.abs(fft_dim3))
# plt.title('FFT of Latent Vector 3')
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')

# plt.tight_layout()
# plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/latent_23_3dim.png')

# # 
# fft_segidx = np.fft.fft(segidx,axis=0)
# fft_ridx = np.fft.fft(ridx,axis=0)
# fft_hidx = np.fft.fft(hidx,axis=0)

# frequency_segidx = np.fft.fftfreq(segidx.size,d=7e-3)
# frequency_ridx = np.fft.fftfreq(ridx.size,d=7e-3)
# frequency_hidx = np.fft.fftfreq(hidx.size,d=7e-3)

# plt.subplot(3, 1, 1)
# plt.plot(frequency_segidx, np.abs(fft_segidx))
# plt.title('FFT of Segidx')
# plt.xlabel('Frequency')
# plt.xlim(-2,2)
# plt.ylabel('Amplitude')

# plt.subplot(3, 1, 2)
# plt.plot(frequency_ridx, np.abs(fft_ridx))
# plt.title('FFT of Ridx')
# plt.xlim(-2,2)
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')

# plt.subplot(3, 1, 3)
# plt.plot(frequency_hidx, np.abs(fft_hidx))
# plt.xlim(-2,2)
# plt.title('FFT of Hidx')
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')

# plt.tight_layout()
# plt.show()
# # plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/Seg_Ridx_Hidx_fft_3.png')


# training_curve_linear_feature = np.load('training_curve_linear_feature.npy')
# training_curve_linear_init = np.load('training_curve_linear_init.npy')
# training_curve_nonlinear_feature = np.load('training_curve_nonlinear_feature.npy')
# training_curve_noninear_init = np.load('training_curve_nonlinear_init.npy')

# plt.figure()
# plt.plot(training_curve_nonlinear_feature, label='nonlinear_feature')
# plt.plot(training_curve_linear_feature, label='linear_feature')
# plt.plot(training_curve_linear_init, label='linear_init')
# plt.plot(training_curve_noninear_init, label='nonlinear_init')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()




