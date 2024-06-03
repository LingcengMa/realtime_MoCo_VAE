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
from network import VAE, CVAE
# from Vae1 import vae_test2, vae_test1, vae_test3, vae_test4, vae_test5, vae_test6, vae_test8
#from models import VariationalAutoencoder_hc, VariationalAutoencoder_hc_m, VariationalAutoencoder_linear4, VariationalAutoencoder_nonlinear4
import torchvision
import numpy as np
import scipy
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200


path = os.path.dirname(os.path.realpath(__file__))
data_file = os.path.join(path, 'assets/recon_output_CVAEBP_25_w_eddy.npy')
data_w_eddy = np.load(data_file)
data_file = os.path.join(path, 'assets/recon_output_CVAEBP_25_wo_eddy_pausecontrast.npy')
data_wo_eddy_pausecontrast = np.load(data_file)
data_file = os.path.join(path, 'assets/recon_output_CVAEBP_25_wo_eddy.npy')
data_wo_eddy = np.load(data_file)
data_file = os.path.join(path, 'assets/recon_output_CVAEBP_25_wo_eddy_pausemotion.npy')
data_wo_eddy_pausemotion = np.load(data_file)

# Save as .mat file
scipy.io.savemat('ImportToMatlab_w_eddy.mat', {'data': data_w_eddy})
scipy.io.savemat('ImportToMatlab_wo_eddy.mat', {'data': data_wo_eddy})
scipy.io.savemat('ImportToMatlab_wo_eddy_pausecontrast.mat', {'data': data_wo_eddy_pausecontrast})
scipy.io.savemat('ImportToMatlab_wo_eddy_pausemotion.mat', {'data': data_wo_eddy_pausemotion})
