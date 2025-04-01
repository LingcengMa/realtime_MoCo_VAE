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
from tqdm import tqdm
from network import VAE, CVAE, CVAE_BP, CVAE_BP_HT
import torchvision
import mat73
import numpy as np
import scipy
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200
from sklearn.feature_selection import mutual_info_regression
import scipy.stats as stats
import statsmodels.api as sm
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.io import savemat

torch.autograd.set_detect_anomaly(True)

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
path = os.path.dirname(os.path.realpath(__file__))

# load data and conditional information
data_file = "/home/xfang/projects/CVAE_SMS/SMS_data/ExtractToPython_0618_2_run1.mat"
mat_data = mat73.loadmat(data_file)
mat_data = mat_data["ExtractToPython"]
Phi_rt_init = mat_data["Phi_rt_small"]
Segidx = mat_data["Segidx"].reshape(-1, 1)
wallClock = mat_data["wallClock"].reshape(-1, 1)
thetas = mat_data["thetas"].reshape(-1, 1)

if "flipAngle" in mat_data:
    flipAngles = mat_data["flipAngle"].reshape(-1, 1)
    IFdualFA = True
else:
    flipAngles = np.zeros_like(thetas)
    
### fixing wallclock ###
wallClock = (wallClock - 1) % 5 + 1

### processing mat data ###
Segidx_mu = (Segidx - Segidx.mean()) / Segidx.std()
wallClock_mu = (wallClock - wallClock.mean()) / wallClock.std()
thetas_mu = (thetas - thetas.mean()) / thetas.std()
flipAngles_mu = (flipAngles - flipAngles.mean()) / flipAngles.std()

wallClock_mu = torch.from_numpy(wallClock_mu).to(device)
Segidx_mu = torch.from_numpy(Segidx_mu).to(device)
thetas_mu = torch.from_numpy(thetas_mu).to(device).to(torch.float32)
flipAngles_mu = torch.from_numpy(flipAngles_mu).to(device)
theta0 = torch.zeros_like(thetas_mu).to(torch.float32)

### resample data for mapping

### Concatenating the data for conditional input ###
t1t2_normalized = torch.cat([Segidx_mu, wallClock_mu], dim=1).to(torch.float32)
t1t2theta = torch.cat([Segidx_mu, wallClock_mu, thetas_mu], dim=1).to(torch.float32)
t1t2theta0 = torch.cat([Segidx_mu, wallClock_mu, theta0], dim=1).to(torch.float32)
t1t2fatheta = torch.cat(
    [Segidx_mu, wallClock_mu, flipAngles_mu, thetas_mu], dim=1
).to(torch.float32)
t1t2fatheta0 = torch.cat(
    [Segidx_mu, wallClock_mu, flipAngles_mu, theta0], dim=1
).to(torch.float32)

Phi_rt_init = Phi_rt_init / Phi_rt_init[:, 0].std()  # Normalize whole
# Phi_rt_init = (Phi_rt_init - Phi_rt_init.mean(axis=1, keepdims=True)) / Phi_rt_init.std(axis=1, keepdims=True) #Normalize each feature
Phi_rt_init_real = Phi_rt_init.real
Phi_rt_init_imag = Phi_rt_init.imag
Phi_rt_init_con = np.concatenate((Phi_rt_init_real, Phi_rt_init_imag))
Phi_rt_init_con = Phi_rt_init_con.transpose()
Phi_rt_init = torch.from_numpy(Phi_rt_init_con).float().to(device)

feature_size = Phi_rt_init.size()[1]
model = CVAE_BP_HT(feature_size, latent_dims=2, class_size=4).to(device)
model.load_state_dict(torch.load('/home/xfang/projects/CVAE_SMS/SMS_models_kld/model_ExtractToPython_0618_2_run1.pth'))

recon_output_ec, latent_space_mu, latent_space_logvar = model(Phi_rt_init, t1t2fatheta)
recon_output_ec = model.decode(latent_space_mu, t1t2fatheta).to("cpu").detach().numpy()
recon_output = model.decode(latent_space_mu, t1t2fatheta0).to("cpu").detach().numpy()
latent_space = model.reparameterize(latent_space_mu, latent_space_logvar)
latent_space = latent_space.to("cpu").detach().numpy()
latent_space_mu = latent_space_mu.to("cpu").detach().numpy()

resp = latent_space_mu[:, 0]
resp_min = np.percentile(resp, 10)
resp_max = np.percentile(resp, 90)
counts, bins, patches = plt.hist(resp, bins=100, edgecolor="black")
resp0_index = np.argmax(counts)
bin_range = (bins[resp0_index], bins[resp0_index + 1])
resp0 = (bin_range[0] + bin_range[1]) / 2

# cardiac cycle

cardiac1 = latent_space_mu[:, 1]
cardiac2 = latent_space_mu[:, 2]

def average_cycle(cardiac1, cardiac2):
    card_amplitude = np.sqrt(cardiac1**2 + cardiac2**2)
    card_phase = np.arctan2(cardiac2, cardiac1)
    left_phase = card_phase - 2*np.pi
    right_phase = card_phase + 2*np.pi
    padded_amplitude = np.concatenate((card_amplitude, card_amplitude, card_amplitude))
    padded_phase = np.concatenate((left_phase, card_phase, right_phase))
    lowess = sm.nonparametric.lowess
    smoothed_cardiac = lowess(padded_amplitude, padded_phase, frac=0.3)
    smoothed_phase = smoothed_cardiac[cardiac1.shape[0]:(2*cardiac1.shape[0]-1), 0]
    smoothed_amplitude = smoothed_cardiac[cardiac1.shape[0]:(2*cardiac1.shape[0]-1), 1]
    min_index = np.argmin(smoothed_amplitude)  # Index of min amplitude
    min_phase = smoothed_phase[min_index]  # Corresponding phase
    phase0 = min_phase
    unique_phase, unique_indices = np.unique(smoothed_phase, return_index=True)
    unique_amplitude = smoothed_amplitude[unique_indices]  # Keep only unique amplitude
    amplitude_function = interp1d(unique_phase, unique_amplitude, kind="cubic", fill_value="extrapolate")
    return phase0, amplitude_function

phase0, amplitude_function = average_cycle(cardiac1, cardiac2)

cbins = 50
rbins = 20
contrast_num = 3600
module_num = 10
contrast_sub = contrast_num // module_num
conditional_input = 4
Segidx_excerpt = Segidx_mu[:contrast_num//module_num,:].to("cpu").detach().numpy()
axis = np.linspace(1, contrast_num/module_num, contrast_num//module_num, endpoint=False)
axis_interp = np.linspace(1, contrast_num/module_num, 2*contrast_num//module_num, endpoint=False) 
f = interp1d(axis, Segidx_excerpt, axis=0, kind='linear', fill_value='extrapolate')
Segidx_interp_np = f(axis_interp)
Segidx_interp = torch.tensor(Segidx_interp_np, dtype=torch.float32).repeat(10,1).to(device)
conditional_3 = t1t2fatheta0[:contrast_num,1:4].unsqueeze(0).permute(0, 2, 1)
conditional_3_interp = F.interpolate(conditional_3, size=2*contrast_num, mode='linear', align_corners=True)
conditional_3_interp = conditional_3_interp.permute(0, 2, 1).squeeze(0)
conditional_interp = torch.cat((Segidx_interp, conditional_3_interp), dim=1)

latent_space_mu_fitting = torch.ones(contrast_num*2*cbins*rbins, 3).to(device)
resp_bins = torch.linspace(resp_min, resp_max, rbins)
latent_space_mu_fitting[:,0] = resp_bins.repeat_interleave(contrast_num*2*cbins)
card_phase_values = torch.linspace(phase0, phase0+2*np.pi, cbins)
card_phase_values = (card_phase_values + np.pi) % (2 * np.pi) - np.pi
card_phase_bins = card_phase_values.repeat_interleave(contrast_num*2).repeat(rbins)
card_amplitude_bins = amplitude_function(card_phase_bins)
latent_space_mu_fitting[:,1] = torch.from_numpy(card_amplitude_bins) * torch.cos(card_phase_bins)
latent_space_mu_fitting[:,2] = torch.from_numpy(card_amplitude_bins) * torch.sin(card_phase_bins)
condition_fitting = conditional_interp.repeat(cbins*rbins,1).to(device)
phi4fitting = model.decode(latent_space_mu_fitting,condition_fitting).to("cpu").detach().numpy()

savemat(
    "/home/xfang/Data/Multitasking_SMS/Human_SMS_20210618_2/2D_SMS_DATA/python_0618_2_run1.mat",
    {
        "realtime": recon_output,
        "phi4fitting": phi4fitting,
        "latentspace": latent_space_mu,
        "endofdiastole": phase0,
    },
)

