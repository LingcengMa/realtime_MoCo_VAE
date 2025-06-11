import os
import sys
import torch
# torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils
import torch.distributions
import torch.optim as optim
import mat73
from pathlib import Path
from tqdm import tqdm
from network import Decoder
from torchvision import datasets
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.io import savemat

plt.rcParams["figure.dpi"] = 200
torch.autograd.set_detect_anomaly(True)

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# data_init
path = os.path.dirname(os.path.realpath(__file__))

def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + 1e-9 * KLD  # was 1e-9 for T1-T2 single-FA single-slice (prelim data)

def train_model(
    model, optimizer, max_epochs, epoch_plot_step, Phi_rt_init, conditional_input
):
    losses = []
    NRMSE = []
    total = (Phi_rt_init**2).sum()
    pbar = tqdm(total=max_epochs, desc="Initializing")
    for epoch in range(max_epochs):
        recon_x = model(Phi_rt_init, conditional_input)
        optimizer.zero_grad()
        loss = loss_function(recon_x, Phi_rt_init, 0, 0)
        loss.backward()
        optimizer.step()
        if (epoch % epoch_plot_step) == (epoch_plot_step - 1):
            train_loss = loss.detach().cpu().numpy() / Phi_rt_init.shape[0]
            losses.append(train_loss)
            mse = ((recon_x - Phi_rt_init) ** 2).sum()
            nrmse_ = torch.sqrt(mse / total)
            NRMSE.append(nrmse_.detach().cpu().numpy())
            pbar.set_description(f"train_loss: {train_loss:.3e}, NRMSE: {nrmse_:.4f}")
            pbar.update(epoch_plot_step)
    pbar.close()
    return losses, NRMSE, recon_x

def train_decoder(
    model, optimizer, max_epochs, epoch_plot_step, Phi_rt_init, latent_input, conditional_input
):
    losses = []
    NRMSE = []
    total = (Phi_rt_init**2).sum()
    pbar = tqdm(total=max_epochs, desc="Initializing")
    for epoch in range(max_epochs):
        recon_x = model(latent_input, conditional_input)
        optimizer.zero_grad()
        loss = F.mse_loss(recon_x, Phi_rt_init)
        loss.backward()
        optimizer.step()
        if (epoch % epoch_plot_step) == (epoch_plot_step - 1):
            train_loss = loss.detach().cpu().numpy() / Phi_rt_init.shape[0]
            losses.append(train_loss)
            mse = ((recon_x - Phi_rt_init) ** 2).sum()
            nrmse_ = torch.sqrt(mse / total)
            NRMSE.append(nrmse_.detach().cpu().numpy())
            pbar.set_description(f"train_loss: {train_loss:.3e}, NRMSE: {nrmse_:.4f}")
            pbar.update(epoch_plot_step)
    pbar.close()
    return losses, NRMSE, recon_x


data_folder = "/home/xfang/Data/Multitasking_SMS/Phantom_VIDA_20211012/2D_RAW_DATA"
model_folder = "./SMS_models"
file_name = 'Phantom_VIDA_20211012_ExtractToPython.mat'

mat_path = os.path.join(data_folder, file_name)
file_name = file_name.split(".")[0]
print(f"\nProcessing file: {file_name}")

### Loading data ###
mat_data = mat73.loadmat(mat_path)
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

### Concatenating the data for conditional input ###
t1t2_normalized = torch.cat([Segidx_mu, wallClock_mu], dim=1).to(torch.float32)
t1t2theta = torch.cat([Segidx_mu, wallClock_mu, thetas_mu], dim=1).to(torch.float32)
t1t2theta0 = torch.cat([Segidx_mu, wallClock_mu, theta0], dim=1).to(torch.float32)
t1t2fa = torch.cat([Segidx_mu, wallClock_mu, flipAngles_mu], dim=1).to(
    torch.float32
)
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

output_size = Phi_rt_init.size()[1]
latent_input = torch.zeros(25920,3).to(device)
model = Decoder(latent_dims=3, conditional_size=4, feature_size = output_size).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
max_epochs = 30000
epoch_plot_step = 10
checkpoint = 10000

print(f"Training on Data File: {file_name}")
losses, NRMSE, recon_x = train_decoder(
    model,
    optimizer,
    max_epochs,
    epoch_plot_step,
    Phi_rt_init,
    latent_input,
    t1t2fatheta
    )
    # Save the model after training on the current dataset
torch.save(
    model,
    os.path.join(model_folder, f"model_{file_name}.pt"),
    )
recon_x_np = recon_x.to('cpu').detach().numpy()
# load saved model
# model.load_state_dict(torch.load('SMS_models/model_Phantom_VIDA_20211012_ExtractToPython.pt'))
# model.eval()

# recon_x_np = model(latent_input, t1t2fatheta).to('cpu').detach().numpy()

latent_input_mapping = torch.zeros(7200, 3).to(device)
cbins = 1
rbins = 1
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
conditional_interp = torch.cat((Segidx_interp, conditional_3_interp), dim=1).to(device)
phi4fitting = model(latent_input_mapping,conditional_interp).to("cpu").detach().numpy()


savemat(
    "/home/xfang/Data/Multitasking_SMS/Phantom_VIDA_20211012/2D_RAW_DATA/python_phantom.mat",
    {
        "realtime": recon_x_np,
        "phi4fitting": phi4fitting,
    },
)

print(f"Model saved as model_{file_name}.pt") 

