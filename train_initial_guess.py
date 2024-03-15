import os
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torch.optim as optim
from vae import VariationalAutoencoder
from models import VariationalAutoencoder_hc
from BasicVae import VAE
import torchvision
import numpy as np
import scipy
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

#Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#data
data_file = os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'realtime_GT_imgs.mat')
mat = scipy.io.loadmat(data_file)
data = mat['Phi_rt_init'] 
data_real = data.real
data_imag = data.imag
data_con = np.concatenate((data_real, data_imag))
data_con = data_con.transpose()
data = torch.from_numpy(data_con).float()

#data_feature
data_file= os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'feature.mat')
mat = scipy.io.loadmat(data_file)
Phi_rt_feature = mat['Phi_rt_feature'] 
Phi_rt_feature_real = Phi_rt_feature.real
Phi_rt_feature_imag = Phi_rt_feature.imag
Phi_rt_feature_con = np.concatenate((Phi_rt_feature_real, Phi_rt_feature_imag))
Phi_rt_feature_con = Phi_rt_feature_con.transpose()
Phi_rt_feature = torch.from_numpy(Phi_rt_feature_con).float()

# print(data_con.shape)

seg_file = os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'TI.mat')
segidx = scipy.io.loadmat(seg_file)
segidx = segidx['Segidx']

binning_file = os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'binning_GT.mat')
binning_file = scipy.io.loadmat(binning_file)
hidx = binning_file['Hidx'] # Cardiac
ridx = binning_file['Ridx'] # Respiratory
ridx_mu = (ridx - ridx.mean())/ridx.std()
ridx_mu = torch.from_numpy(ridx_mu).float()
hidx_mu = (hidx - hidx.mean())/hidx.std()
hidx_mu = torch.from_numpy(hidx_mu).float()
segidx_mu = (segidx - segidx.mean())/segidx.std()

def custom_initialization(model, guess1, guess2):
    with torch.no_grad():
        # Extract weights for the latent mean and log variance layers
        weight_mean, bias_mean = model.encoder.linear5.weight, model.encoder.linear5.bias
        weight_logvar, bias_logvar = model.encoder.linear6.weight, model.encoder.linear6.bias

        # Set the specific latent variables with your initial guesses
        weight_mean[:, 2] = guess1
        weight_logvar[:, 2] = 0
        
        weight_mean[:, 3] = guess2
        weight_logvar[:, 3] = 0

vae = VariationalAutoencoder_hc(latent_dims=4)

optimizer = torch.optim.SGD(vae.parameters(), lr=0.01)
criterion = nn.MSELoss()

num_epochs = 20

for epoch in range(num_epochs):
    for idx, x in enumerate(data):
        custom_initialization(vae, hidx_mu, ridx_mu)
        optimizer.zero_grad()
        recon_x = vae(x,idx)
        loss = criterion(recon_x, x)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')




