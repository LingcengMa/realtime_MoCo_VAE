import os
import sys
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils
import torch.distributions
import torch.optim as optim
import time
from tqdm import tqdm
from network import VAE, CVAE
# from Vae1 import vae_test2, vae_test1, vae_test3, vae_test4, vae_test5, vae_test6, vae_test8
#from models import VariationalAutoencoder_hc, VariationalAutoencoder_hc_m, VariationalAutoencoder_linear4, VariationalAutoencoder_nonlinear4
import torchvision
import numpy as np
import scipy
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from sklearn.feature_selection import mutual_info_regression
from torch.utils.tensorboard import SummaryWriter
#Configuration
device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# MI = MutualInformation(num_bins=256, sigma=0.4, normalize=True).to(device)

#data_init

path = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(path, 'binning')

if True: #to improve: do this with a flag
    data_file = os.path.join(data_folder, 'realtime_GT_imgs.mat')
    mat = scipy.io.loadmat(data_file)
    Phi_rt_init = mat['Phi_rt_init'] 
else :
    data_file = os.path.join(data_folder, 'feature.mat')
    mat = scipy.io.loadmat(data_file)
    Phi_rt_init = mat['Phi_rt_feature']

Phi_rt_init = Phi_rt_init / Phi_rt_init[:,0].std() #Normalize whole
# Phi_rt_init = (Phi_rt_init - Phi_rt_init.mean(axis=1, keepdims=True)) / Phi_rt_init.std(axis=1, keepdims=True) #Normalize each feature
Phi_rt_init_real = Phi_rt_init.real
Phi_rt_init_imag = Phi_rt_init.imag
Phi_rt_init_con = np.concatenate((Phi_rt_init_real, Phi_rt_init_imag))
Phi_rt_init_con = Phi_rt_init_con.transpose()
Phi_rt_init = torch.from_numpy(Phi_rt_init_con).float().to(device)

# print(data_con.shape)

seg_file = os.path.join(data_folder, 'TI.mat')
segidx = scipy.io.loadmat(seg_file)
segidx = segidx['Segidx']

binning_file = os.path.join(data_folder, 'binning_GT.mat')
binning_file = scipy.io.loadmat(binning_file)
hidx = binning_file['Hidx'] # Cardiac
ridx = binning_file['Ridx'] # Respiratory

# create simulated t2 prep signal
array_1 = np.ones(360)
array_2 = 2 * array_1
array_3 = 3 * array_1
array_4 = 4 * array_1
array_5 = 5 * array_1
array = np.concatenate((array_1, array_2, array_3),axis=0)
array = np.concatenate((array, array_4, array_5),axis=0)
array = np.tile(array, 7)

ridx_mu = (ridx - ridx.mean())/ridx.std()
hidx_mu = (hidx - hidx.mean())/hidx.std()
segidx_mu = (segidx - segidx.mean())/segidx.std()
ramp_latent_mu = (array-array.mean())/array.std()

ramp_latent_mu = torch.from_numpy(ramp_latent_mu).unsqueeze(1).to(device)
segidx_mu  = torch.from_numpy(segidx_mu).to(device)
t1t2_normalized = torch.cat([segidx_mu,ramp_latent_mu], dim=1).to(torch.float32)

class CustomDataset(Dataset):
    def __init__(self, data_feature, data_t1t2):
        self.data = data_feature
        self.t1t2 = data_t1t2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the data and t1t2 for the given index
        data_sample = self.data[index]
        t1t2_sample = self.t1t2[index]

        return data_sample, t1t2_sample

train_dataset = CustomDataset(Phi_rt_init, t1t2_normalized)

########### training ###########
latent_dims = 2
class_size = 2
feature_size = 96
model = CVAE(feature_size, latent_dims, class_size).to(device)
max_epochs = 5

optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

max_epochs = 1
losses= []
NRMSE = []

print("training on", device)
total = (Phi_rt_init**2).sum()
for epoch in range(max_epochs):
    train_loss, nrmse_, start = 0.0, 0.0, time.time()
    mse = 0
    for idx, (x,t1t2) in tqdm(enumerate(train_dataset),desc=f"Epoch {epoch + 1}/{max_epochs}"):
    # tqdm(train_dataset,desc=f"Epoch {epoch + 1}/{max_epochs}"):
        x, t1t2 = x.to(device), t1t2.to(device)
        x = x.unsqueeze(0)
        t1t2 = t1t2.unsqueeze(0)
        recon_x, mu, logvar = model(x, t1t2)
        optimizer.zero_grad()
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()
        mse += ((recon_x - x)**2).sum()
        train_loss += loss
        optimizer.step()
    train_loss /= Phi_rt_init.shape[0]
    nrmse_ = torch.sqrt(mse / total)
    NRMSE.append(nrmse_.detach().numpy())
    losses.append(train_loss.detach().numpy())
    print('epoch %d, train loss %.4f , NRMSE %.4f , time %.1f sec'
          % (epoch+1, train_loss, nrmse_, time.time() - start))

# Plot loss functions
fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column, optional figure size
# First subplot for Training Loss
axs[0].plot(losses, label='Training Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training Loss Curve')
axs[0].legend()
# Second subplot for NRMSE
axs[1].plot(NRMSE, label='NRMSE')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('NRMSE')
axs[1].set_title('NRMSE Curve')
axs[1].legend()
plt.tight_layout()
plt.savefig(os.path.join('./lossGraphs', 'losses_test.png'))

print("Training complete")

recon_output, latent_space_mu, latent_space_logvar  = model(Phi_rt_init, t1t2_normalized)
recon_output = recon_output.to('cpu').detach().numpy()
np.save('./assets/recon_output_test.npy', recon_output)
latent_space = model.reparameterize(latent_space_mu,latent_space_logvar)
latent_space = latent_space.to('cpu').detach().numpy()
np.save('./assets/latent_space_test.npy', latent_space)

# Some plots. These should be in a different file at some point
fig, axs = plt.subplots(4, 2, figsize=(10, 10))
axs = axs.flatten()  # Flatten the 2x2 array of axes for easy indexing
for i in range(4):
    axs[i].plot(Phi_rt_init[:1800, i], label='Phi_rt_init Col ' + str(i + 1))
    axs[i].plot(recon_output[:1800, i], label='Recon_output Col ' + str(i + 1))
    axs[i].legend(loc="upper right")
for i in range(2):
    axs[4+i].plot(latent_space[:1800, i], label='Latent Col ' + str(i + 1))
    axs[4+i].legend(loc="upper right")
for i in range(2):
    axs[6+i].plot(t1t2_normalized[:1800, i], label='Known Latent Col ' + str(i + 1))
    axs[6+i].legend(loc="upper right")
plt.tight_layout()
plt.savefig('./plots/4x2_subplots.png')