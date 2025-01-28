import os
import torch

torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
from vae import VariationalAutoencoder
import torchvision
import numpy as np
import scipy
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200
import mat73

# data
recon_output = np.load(
    "/home/xfang/projects/vae_temporal_analysis/assets/recon_output_CVAEBPHT_wo_eddy.npy"
)
recon_output_ec = np.load(
    "/home/xfang/projects/vae_temporal_analysis/assets/recon_output_CVAEBPHT_w_eddy.npy"
)
mat_data = mat73.loadmat(
    "/home/xfang/projects/CVAE_SMS/data_test/ExtractToPython_patient.mat"
)
mat_data = mat_data["ExtractToPython"]
Phi_rt_init = mat_data["Phi_rt_small"]
Phi_rt_init = Phi_rt_init / Phi_rt_init[:, 0].std()
Phi_rt_init_real = Phi_rt_init.real
Phi_rt_init_imag = Phi_rt_init.imag
Phi_rt_init_con = np.concatenate((Phi_rt_init_real, Phi_rt_init_imag))
Phi_rt_init_con = Phi_rt_init_con.transpose()

# output plots
fig, axs = plt.subplots(4, 2, figsize=(20, 20))
axs = axs.flatten()  # Flatten the 2x2 array of axes for easy indexing
for i in range(4):
    axs[i].plot(Phi_rt_init_con[:1800, i], label=f"Phi_rt_init Col {i+1}")
    axs[i].plot(
        recon_output_ec[:1800, i], label=f"Recon_output w/eddy current Col {i+1}"
    )
    axs[i].legend(loc="upper right")
for i in range(4):
    axs[i + 4].plot(Phi_rt_init_con[:1800, i], label=f"Phi_rt_init Col {i+1}")
    axs[i + 4].plot(recon_output[:1800, i], label=f"Recon_output Col {i+1}")
    axs[i + 4].legend(loc="upper right")
plt.tight_layout()
fig.savefig("/home/xfang/Data/ISMRM25/Eddy_Current_Correction.png")


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
