import os
import sys
import torch

torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils
import torch.distributions
import torch.optim as optim
import h5py
import mat73
import time
import torchaudio
from tqdm import tqdm
from network import VAE, CVAE, CVAE_BP, CVAE_BP_HT

# from Vae1 import vae_test2, vae_test1, vae_test3, vae_test4, vae_test5, vae_test6, vae_test8
# from models import VariationalAutoencoder_hc, VariationalAutoencoder_hc_m, VariationalAutoencoder_linear4, VariationalAutoencoder_nonlinear4
import torchvision
import numpy as np
import scipy
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200
from sklearn.feature_selection import mutual_info_regression
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
torch.autograd.set_detect_anomaly(True)
# Configuration
# device = 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"

latent_space = np.load(
    "/home/xfang/projects/vae_temporal_analysis/assets/latent_space_cv.npy"
)
resp_signal = latent_space[:, 0]
card_signal = np.sqrt(latent_space[:, 1] ** 2 + latent_space[:, 2] ** 2)

num_bins_resp = 6
num_bins_card = 21

resp_signal_binned = np.digitize(
    resp_signal, np.linspace(resp_signal.min(), resp_signal.max(), num_bins_resp)
)
card_signal_binned = np.digitize(
    card_signal, np.linspace(card_signal.min(), card_signal.max(), num_bins_card)
)

scipy.io.savemat(
    "binned_motion_cv.mat", {"Ridx": resp_signal_binned, "Hidx": card_signal_binned}
)

print(resp_signal_binned.shape)
#
plt.figure(figsize=(12, 6))
plt.plot(resp_signal_binned[:1800], label="resp bins", color="blue", alpha=0.6)
plt.title("Resp bins")
plt.xlabel("Time")
plt.ylabel("Resp signal")
plt.legend()
plt.savefig("/home/xfang/projects/vae_temporal_analysis/resp_bins.png")

plt.figure(figsize=(12, 6))
plt.plot(card_signal_binned[:1800], label="card bins", color="blue", alpha=0.6)
plt.title("Card bins")
plt.xlabel("Time")
plt.ylabel("Card signal")
plt.legend()
plt.savefig("/home/xfang/projects/vae_temporal_analysis/card_bins.png")
