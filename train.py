import os
import sys
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils
import torch.distributions
import torch.optim as optim
import h5py
import mat73
import time
from tqdm import tqdm
from network import VAE, CVAE, CVAE_BP
# from Vae1 import vae_test2, vae_test1, vae_test3, vae_test4, vae_test5, vae_test6, vae_test8
#from models import VariationalAutoencoder_hc, VariationalAutoencoder_hc_m, VariationalAutoencoder_linear4, VariationalAutoencoder_nonlinear4
import torchvision
import numpy as np
import scipy
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from sklearn.feature_selection import mutual_info_regression
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
torch.autograd.set_detect_anomaly(True)
#Configuration
# device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# MI = MutualInformation(num_bins=256, sigma=0.4, normalize=True).to(device)

#data_init

path = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(path, 'binning')

if False: #to improve: do this with a flag
    data_file = os.path.join(data_folder, 'realtime_GT_imgs.mat')
    mat = scipy.io.loadmat(data_file)
    Phi_rt_init = mat['Phi_rt_init'] 
else :
    data_file = '/home/xfang/Data/stemberg_donna/ExtractToPython.mat'
    mat = mat73.loadmat(data_file)
    mat_data =mat['ExtractToPython']
    Phi_rt_init = mat_data['Phi_rt_small']
    Segidx = mat_data['Segidx'].reshape(-1, 1)
    wallClock = mat_data['wallClock'].reshape(-1, 1)
    thetas = mat_data['thetas']
    thetas = thetas.transpose().reshape(-1, 1)
    
    # hidx = mat['Hidx'] # Cardiac
    # ridx = mat['Ridx'] # Respiratory
    # thetas = mat['thetas'] # Thetas

Phi_rt_init = Phi_rt_init / Phi_rt_init[:,0].std() #Normalize whole
# Phi_rt_init = (Phi_rt_init - Phi_rt_init.mean(axis=1, keepdims=True)) / Phi_rt_init.std(axis=1, keepdims=True) #Normalize each feature
Phi_rt_init_real = Phi_rt_init.real
Phi_rt_init_imag = Phi_rt_init.imag
Phi_rt_init_con = np.concatenate((Phi_rt_init_real, Phi_rt_init_imag))
Phi_rt_init_con = Phi_rt_init_con.transpose()
Phi_rt_init = torch.from_numpy(Phi_rt_init_con).float().to(device)
np.save('./assets/Phi_init_python.npy', Phi_rt_init_con)

# print(data_con.shape)

# seg_file = os.path.join(data_folder, 'TI.mat')
# segidx = scipy.io.loadmat(seg_file)
# segidx = segidx['Segidx']

# binning_file = os.path.join(data_folder, 'binning_GT.mat')
# binning_file = scipy.io.loadmat(binning_file)
# hidx = binning_file['Hidx'] # Cardiac
# ridx = binning_file['Ridx'] # Respiratory

# create simulated t2 prep signal
# array_1 = np.ones(360)
# array_2 = 2 * array_1
# array_3 = 3 * array_1
# array_4 = 4 * array_1
# array_5 = 5 * array_1
# array = np.concatenate((array_1, array_2, array_3),axis=0)
# array = np.concatenate((array, array_4, array_5),axis=0)
# array = np.tile(array, 7)

# ridx_mu = (ridx - ridx.mean())/ridx.std()
# hidx_mu = (hidx - hidx.mean())/hidx.std()
Segidx_mu = (Segidx - Segidx.mean())/Segidx.std()
# ramp_latent_mu = (array-array.mean())/array.std()
wallClock_mu = (wallClock - wallClock.mean())/wallClock.std()
thetas_mu = (thetas - thetas.mean())/thetas.std()


# ramp_latent_mu = torch.from_numpy(ramp_latent_mu).unsqueeze(1).to(device)
wallClock_mu = torch.from_numpy(wallClock_mu).to(device)
Segidx_mu  = torch.from_numpy(Segidx_mu).to(device)
thetas_mu = torch.from_numpy(thetas_mu).to(device)
theta0 = torch.zeros_like(thetas_mu)
t1t2_normalized = torch.cat([Segidx_mu, wallClock_mu], dim=1).to(torch.float32)
t1t2theta = torch.cat([Segidx_mu, wallClock_mu, thetas_mu], dim=1).to(torch.float32)
t1t2theta0 = torch.cat([Segidx_mu, wallClock_mu, theta0], dim=1).to(torch.float32) 

########### training ###########
latent_dims = 2
class_size = 2
feature_size = Phi_rt_init.size()[1]

# model = CVAE(feature_size, latent_dims, class_size).to(device)

# optimizer = optim.Adam(model.parameters(), lr=1e-2)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + 1e-9*KLD

# max_epochs = 10000
# epoch_plot_step = 10
# losses= []
# NRMSE = []

# print("training on", device)
# total = (Phi_rt_init**2).sum()
# pbar = tqdm(total=max_epochs, desc='Initializing')
# for epoch in range(max_epochs):
#     recon_x, mu, logvar = model(Phi_rt_init, t1t2_normalized)
#     optimizer.zero_grad()
#     loss = loss_function(recon_x, Phi_rt_init, mu, logvar)
#     loss.backward()
#     optimizer.step()
#     if (epoch % epoch_plot_step) == (epoch_plot_step - 1) :
#         train_loss = loss.detach().cpu().numpy()/Phi_rt_init.shape[0]
#         losses.append(train_loss)
#         mu, logvar = model.encode(Phi_rt_init, t1t2_normalized)
#         recon_x = model.decode(mu, t1t2_normalized) #NO REPARAMETERIZATION
#         mse = ((recon_x - Phi_rt_init)**2).sum()
#         nrmse_ = torch.sqrt(mse / total)
#         NRMSE.append(nrmse_.detach().cpu().numpy())
#         pbar.set_description(f'train_loss: {train_loss:.3e}, NRMSE: {nrmse_:.4f}')
#         pbar.update(epoch_plot_step)
# pbar.close()

# # Plot loss functions
# fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column, optional figure size
# # First subplot for Training Loss
# axs[0].plot((np.arange(len(losses)) +1) * epoch_plot_step, losses, label='Training Loss')
# axs[0].set_xlabel('Epochs')
# axs[0].set_ylabel('Loss')
# axs[0].set_title('Training Loss Curve')
# axs[0].legend()
# # Second subplot for NRMSE
# axs[1].plot((np.arange(len(NRMSE)) + 1) * epoch_plot_step, NRMSE, label='NRMSE')
# axs[1].set_xlabel('Epochs')
# axs[1].set_ylabel('NRMSE')
# axs[1].set_title('NRMSE Curve')
# axs[1].legend()
# plt.tight_layout()
# plt.savefig(os.path.join('./lossGraphs', 'losses_2inf.png'))

# print("Training complete")

# latent_space_mu, latent_space_logvar = model.encode(Phi_rt_init, t1t2_normalized)
# # without theta
# recon_output = model.decode(mu, t1t2_normalized) #NO REPARAMETERIZATION
# recon_output = recon_output.to('cpu').detach().numpy()
# np.save('./assets/recon_output_stemberg_2inf.npy', recon_output)
# latent_space = model.reparameterize(latent_space_mu,latent_space_logvar)
# latent_space = latent_space.to('cpu').detach().numpy()
# np.save('./assets/latent_space_stemberg_2inf.npy', latent_space)

# # Some plots. These should be in a different file at some point
# fig, axs = plt.subplots(4, 2, figsize=(10, 10))
# axs = axs.flatten()  # Flatten the 2x2 array of axes for easy indexing
# for i in range(4):
#     axs[i].plot(Phi_rt_init[:1800, i].cpu(), label='Phi_rt_init Col ' + str(i + 1))
#     axs[i].plot(recon_output[:1800, i], label='Recon_output Col ' + str(i + 1))
#     axs[i].legend(loc="upper right")
# for i in range(2):
#     axs[4+i].plot(latent_space_mu.detach().cpu().numpy()[:1800,i], label='Latent Col ' + str(i + 1))
#     axs[4+i].legend(loc="upper right")
# for i in range(2):
#     axs[6+i].plot(t1t2theta[:1800, i].cpu(), label='Known Latent Col ' + str(i + 1))
#     axs[6+i].legend(loc="upper right")
# # axs[9].plot(t1t2theta[:360,2].cpu(), label='theta')
# plt.tight_layout()
# plt.savefig('./plots/4x2_subplots_stemberg_2inf.png')

##### add thetas = 0 as the third parameter
class_size = 3
model = CVAE_BP(feature_size, latent_dims, class_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

max_epochs = 10000
epoch_plot_step = 10
losses= []
NRMSE = []

print("training on", device)
total = (Phi_rt_init**2).sum()
pbar = tqdm(total=max_epochs, desc='Initializing')
for epoch in range(max_epochs):
    recon_x, mu, logvar = model(Phi_rt_init, t1t2theta)
    mu_filtered = mu
    u1 = mu[:,0]
    u2 = mu[:,1]
    u1_fft = model.to_fft(u1)
    u2_fft = model.to_fft(u2)
    u1_filtered = model.low_pass_filter(u1_fft, 0.67)
    u2_filtered = model.band_pass_filter(u2_fft, 0.67, 3)
    mu_filtered[:,0] = model.from_frequency_domain(u1_filtered)
    mu_filtered[:,1] = model.from_frequency_domain(u2_filtered)
    optimizer.zero_grad()
    loss = loss_function(recon_x, Phi_rt_init, mu_filtered, logvar)
    loss.backward()
    optimizer.step()
    if (epoch % epoch_plot_step) == (epoch_plot_step - 1) :
        train_loss = loss.detach().cpu().numpy()/Phi_rt_init.shape[0]
        losses.append(train_loss)
        mu, logvar = model.encode(Phi_rt_init, t1t2theta)
        mu_filtered = mu
        u1 = mu[:,0]
        u2 = mu[:,1]
        u1_fft = model.to_fft(u1)
        u2_fft = model.to_fft(u2)
        u1_filtered = model.low_pass_filter(u1_fft, 0.67)
        u2_filtered = model.band_pass_filter(u2_fft, 0.67, 3)
        mu_filtered[:,0] = model.from_frequency_domain(u1_filtered)
        mu_filtered[:,1] = model.from_frequency_domain(u2_filtered)
        recon_x = model.decode(mu_filtered, t1t2theta) #NO REPARAMETERIZATION
        mse = ((recon_x - Phi_rt_init)**2).sum()
        nrmse_ = torch.sqrt(mse / total)
        NRMSE.append(nrmse_.detach().cpu().numpy())
        pbar.set_description(f'train_loss: {train_loss:.3e}, NRMSE: {nrmse_:.4f}')
        pbar.update(epoch_plot_step)
pbar.close()

# Plot loss functions
fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column, optional figure size
# First subplot for Training Loss
axs[0].plot((np.arange(len(losses)) +1) * epoch_plot_step, losses, label='Training Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training Loss Curve')
axs[0].legend()
# Second subplot for NRMSE
axs[1].plot((np.arange(len(NRMSE)) + 1) * epoch_plot_step, NRMSE, label='NRMSE')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('NRMSE')
axs[1].set_title('NRMSE Curve')
axs[1].legend()
plt.tight_layout()
plt.savefig(os.path.join('./lossGraphs', 'losses_3inf_theta0.png'))

print("Training complete")

latent_space_mu, latent_space_logvar = model.encode(Phi_rt_init, t1t2theta)
recon_output = model.decode(latent_space_mu, t1t2theta) #NO REPARAMETERIZATION
recon_output = recon_output.to('cpu').detach().numpy()
np.save('./assets/recon_output_stemberg_BPF.npy', recon_output)
latent_space = model.reparameterize(latent_space_mu,latent_space_logvar)
latent_space = latent_space.to('cpu').detach().numpy()
np.save('./assets/latent_space_stemberg_BPF.npy', latent_space)

fig, axs = plt.subplots(5, 2, figsize=(10, 10))
axs = axs.flatten()  # Flatten the 2x2 array of axes for easy indexing
for i in range(4):
    axs[i].plot(Phi_rt_init[:1800, i].cpu(), label='Phi_rt_init Col ' + str(i + 1))
    axs[i].plot(recon_output[:1800, i], label='Recon_output Col ' + str(i + 1))
    axs[i].legend(loc="upper right")
for i in range(2):
    axs[4+i].plot(latent_space_mu.detach().cpu().numpy()[:1800,i], label='Latent Col ' + str(i + 1))
    axs[4+i].legend(loc="upper right")
for i in range(3):
    axs[6+i].plot(t1t2theta0[:1800, i].cpu(), label='Known Latent Col ' + str(i + 1))
    axs[6+i].legend(loc="upper right")
axs[9].plot(t1t2theta[:360,2].cpu(), label='theta')
plt.tight_layout()
plt.savefig('./plots/5x2_subplots_stemberg_3inf_theta0.png')


# #### add thetas as the third parameter
# class_size = 3
# model1 = CVAE(feature_size, latent_dims, class_size).to(device)
# optimizer = optim.Adam(model1.parameters(), lr=1e-2)

# max_epochs = 10000
# epoch_plot_step = 10
# losses= []
# NRMSE = []

# print("training on", device)
# total = (Phi_rt_init**2).sum()
# pbar = tqdm(total=max_epochs, desc='Initializing')
# for epoch in range(max_epochs):
#     recon_x, mu, logvar = model1(Phi_rt_init, t1t2theta)
#     optimizer.zero_grad()
#     loss = loss_function(recon_x, Phi_rt_init, mu, logvar)
#     loss.backward()
#     optimizer.step()
#     if (epoch % epoch_plot_step) == (epoch_plot_step - 1) :
#         train_loss = loss.detach().cpu().numpy()/Phi_rt_init.shape[0]
#         losses.append(train_loss)
#         mu, logvar = model1.encode(Phi_rt_init, t1t2theta)
#         recon_x = model1.decode(mu, t1t2theta) #NO REPARAMETERIZATION
#         mse = ((recon_x - Phi_rt_init)**2).sum()
#         nrmse_ = torch.sqrt(mse / total)
#         NRMSE.append(nrmse_.detach().cpu().numpy())
#         pbar.set_description(f'train_loss: {train_loss:.3e}, NRMSE: {nrmse_:.4f}')
#         pbar.update(epoch_plot_step)
# pbar.close()

# # Plot loss functions
# fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column, optional figure size
# # First subplot for Training Loss
# axs[0].plot((np.arange(len(losses)) +1) * epoch_plot_step, losses, label='Training Loss')
# axs[0].set_xlabel('Epochs')
# axs[0].set_ylabel('Loss')
# axs[0].set_title('Training Loss Curve')
# axs[0].legend()
# # Second subplot for NRMSE
# axs[1].plot((np.arange(len(NRMSE)) + 1) * epoch_plot_step, NRMSE, label='NRMSE')
# axs[1].set_xlabel('Epochs')
# axs[1].set_ylabel('NRMSE')
# axs[1].set_title('NRMSE Curve')
# axs[1].legend()
# plt.tight_layout()
# plt.savefig(os.path.join('./lossGraphs', 'losses_3inf.png'))

# print("Training complete")

# latent_space_mu, latent_space_logvar = model1.encode(Phi_rt_init, t1t2theta)
# # with theta
# recon_output = model1.decode(mu, t1t2theta0) #NO REPARAMETERIZATION
# recon_output = recon_output.to('cpu').detach().numpy()
# np.save('./assets/recon_output_stemberg_3inf_theta0.npy', recon_output)
# latent_space = model1.reparameterize(latent_space_mu,latent_space_logvar)
# latent_space = latent_space.to('cpu').detach().numpy()
# np.save('./assets/latent_space_stemberg_3inf_theta0.npy', latent_space)

# fig, axs = plt.subplots(5, 2, figsize=(10, 10))
# axs = axs.flatten()  # Flatten the 2x2 array of axes for easy indexing
# for i in range(4):
#     axs[i].plot(Phi_rt_init[:1800, i].cpu(), label='Phi_rt_init Col ' + str(i + 1))
#     axs[i].plot(recon_output[:1800, i], label='Recon_output Col ' + str(i + 1))
#     axs[i].legend(loc="upper right")
# for i in range(2):
#     axs[4+i].plot(latent_space_mu.detach().cpu().numpy()[:1800,i], label='Latent Col ' + str(i + 1))
#     axs[4+i].legend(loc="upper right")
# for i in range(3):
#     axs[6+i].plot(t1t2theta[:1800, i].cpu(), label='Known Latent Col ' + str(i + 1))
#     axs[6+i].legend(loc="upper right")
# axs[9].plot(t1t2theta[:360,2].cpu(), label='theta')
# plt.tight_layout()
# plt.savefig('./plots/5x2_subplots_stemberg_3inf_theta0.png')

