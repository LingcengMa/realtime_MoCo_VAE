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
data_file = os.path.join(path, 'SMS/recon_output_0528_run1.npy')
data_wo_eddy = np.load(data_file)
data_file = os.path.join(path, 'SMS/recon_output_resp_0528_run1.npy')
data_resp_phase = np.load(data_file)
data_file = os.path.join(path, 'SMS/recon_output_contrast_0528_run1.npy')
data_contrast = np.load(data_file)
data_file = os.path.join(path, 'SMS/recon_output_card_0528_run1.npy')
data_sin_card = np.load(data_file)


# Save as .mat file
# scipy.io.savemat('ImportToMatlab_HT_ee_ed.mat', {'data': data_})
scipy.io.savemat('/home/xfang/Data/Multitasking_SMS/Human_SMS_20210528/2D_SMS_DATA/ImportToMatlab_0528_run1.mat', {'data': data_wo_eddy})
scipy.io.savemat('/home/xfang/Data/Multitasking_SMS/Human_SMS_20210528/2D_SMS_DATA/ImportToMatlab_resp_0528_run1.mat', {'data': data_resp_phase})
scipy.io.savemat('/home/xfang/Data/Multitasking_SMS/Human_SMS_20210528/2D_SMS_DATA/ImportToMatlab_card_0528_run1.mat', {'data': data_sin_card})
scipy.io.savemat('/home/xfang/Data/Multitasking_SMS/Human_SMS_20210528/2D_SMS_DATA/ImportToMatlab_contrast_0528_run1.mat', {'data': data_contrast})
