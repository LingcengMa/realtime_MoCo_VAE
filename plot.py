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
writer = SummaryWriter()
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
    data_file = '/home/xfang/Data/Tianle_data/20210604_workspace_small.mat'
    mat = scipy.io.loadmat(data_file)
    Phi_rt_init = mat['Phi_rt_init']
    hidx = mat['Hidx'] # Cardiac
    ridx = mat['Ridx'] # Respiratory
    thetas = mat['thetas'] # Thetas

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
# hidx = binning_file['Hidx'] # Cardiac
# ridx = binning_file['Ridx'] # Respiratory

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
thetas = torch.from_numpy(thetas.transpose()).to(device)
t1t2_normalized = torch.cat([segidx_mu,ramp_latent_mu], dim=1).to(torch.float32)
t1t2theta = torch.cat([segidx_mu, ramp_latent_mu, thetas], dim=1).to(torch.float32)
print(t1t2theta.shape)
print(t1t2_normalized.shape)

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
train_dataset_ec = CustomDataset(Phi_rt_init, t1t2theta)

# Plot the joint histogram
plt.hist2d(x, y, bins=(10, 10), cmap=plt.cm.jet)
plt.colorbar(label='Counts in Bin')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Joint Histogram of Two Vectors')
plt.show()