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
from network import CVAE_BP_HT
import numpy as np
import scipy
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200
torch.autograd.set_detect_anomaly(True)

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# getting input and output paths from matlab
input_mat = sys.argv[1]
output_mat = sys.argv[2]

# input_mat = "/mnt/local_raid/xfang/Multitasking_SMS/Human_SMS_20210618_2/2D_SMS_DATA/meas_MID00159_FID30158_2D_Multitasking_Multiband_2FA_HYPT2Prep+IR_run1_20250715T232321/ExtractToPython_20210618_2_run1.mat"


# Define the loss function for the CVAE
# The loss function combines reconstruction loss and KL divergence
# The reconstruction loss is typically MSE or binary cross-entropy
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + 1e-9 * KLD  # was 1e-9 for T1-T2 single-FA single-slice (prelim data)


# KLD is imported from Matlab
def loss_function_KLD(recon_x, x, mu, logvar, kld_index):
    BCE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + kld_index * KLD


# data_init
path = os.path.dirname(os.path.realpath(__file__))


def reset_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(
            m.weight, 0, "fan_in", "leaky_relu", torch.cuda.manual_seed_all(8)
        )
        # seed(8) NRMSE at lowest: 0.1974
        # torch.nn.init.kaiming_uniform_(m.weight,0,"fan_in","leaky_relu")
        if m.bias is not None:
            m.bias.data.fill_(0.01)


### training functions ###
def train_model(
    model, optimizer, max_epochs, epoch_plot_step, Phi_rt_init, conditional_input
):
    losses = []
    NRMSE = []
    total = (Phi_rt_init**2).sum()
    pbar = tqdm(total=max_epochs, desc="Initializing")
    for epoch in range(max_epochs):
        recon_x, mu, logvar = model(Phi_rt_init, conditional_input)
        optimizer.zero_grad()
        loss = loss_function(recon_x, Phi_rt_init, mu, logvar)
        loss.backward()
        optimizer.step()
        if (epoch % epoch_plot_step) == (epoch_plot_step - 1):
            train_loss = loss.detach().cpu().numpy() / Phi_rt_init.shape[0]
            losses.append(train_loss)
            mu, logvar = model.encode(Phi_rt_init, conditional_input)
            recon_x = model.decode(mu, conditional_input)  # NO REPARAMETERIZATION
            mse = ((recon_x - Phi_rt_init) ** 2).sum()
            nrmse_ = torch.sqrt(mse / total)
            NRMSE.append(nrmse_.detach().cpu().numpy())
            pbar.set_description(f"train_loss: {train_loss:.3e}, NRMSE: {nrmse_:.4f}")
            pbar.update(epoch_plot_step)
    pbar.close()
    return losses, NRMSE


def train(mat_path):
    print(f"Loading data from: {mat_path}")

    # Load the .mat file (saved with -v7.3 in MATLAB)
    data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    # Access the top-level structure from MATLAB
    ExtractToPython = data["ExtractToPython"]

    # Extract variables (examples below, adapt as needed)
    Phi_rt_init = (
        torch.from_numpy(ExtractToPython.Phi_rt_init.transpose()).float().to(device)
    )
    TrainingCondition = (
        torch.from_numpy(ExtractToPython.TrainingCondition).float().to(device)
    )
    InferenceCondition = (
        torch.from_numpy(ExtractToPython.InferenceCondition).float().to(device)
    )

    lr = ExtractToPython.lr
    NumofEpochs = ExtractToPython.NumofEpochs
    class_size = ExtractToPython.NumofCon
    KLD_index = ExtractToPython.KLDindex
    LPF = ExtractToPython.LPF
    BPFlb = ExtractToPython.BPFlb
    BPFub = ExtractToPython.BPFub
    sampling_rate = ExtractToPython.sampling_rate
    feature_size = Phi_rt_init.size()[1]
    latent_dim = ExtractToPython.NumofLatent

    print("Training setup:")
    print(f"  Phi_rt_init shape: {Phi_rt_init.shape}")
    print(f"  TrainingCondition shape: {TrainingCondition.shape}")
    print(f"  Learning Rate: {lr}, Epochs: {NumofEpochs}")
    print(f"  Class Size: {class_size}, KLD Index: {KLD_index}")
    print(
        f"  LPF: {LPF}, BPFlb: {BPFlb}, BPFub: {BPFub}, Sampling Rate: {sampling_rate}"
    )

    model = CVAE_BP_HT(
        feature_size, latent_dim, class_size, sampling_rate, LPF, BPFlb, BPFub
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.apply(reset_weights)

    epoch_plot_step = 10  # Plot every 10 epochs

    losses = []
    NRMSE = []
    total = (Phi_rt_init**2).sum()
    pbar = tqdm(total=NumofEpochs, desc="Initializing")
    for epoch in range(NumofEpochs):
        recon_x, mu, logvar = model(Phi_rt_init, TrainingCondition)
        optimizer.zero_grad()
        loss = loss_function_KLD(recon_x, Phi_rt_init, mu, logvar, KLD_index)
        loss.backward()
        optimizer.step()
        if (epoch % epoch_plot_step) == (epoch_plot_step - 1):
            train_loss = loss.detach().cpu().numpy() / Phi_rt_init.shape[0]
            losses.append(train_loss)
            mu, logvar = model.encode(Phi_rt_init, TrainingCondition)
            recon_x = model.decode(mu, TrainingCondition)  # NO REPARAMETERIZATION
            mse = ((recon_x - Phi_rt_init) ** 2).sum()
            nrmse_ = torch.sqrt(mse / total)
            NRMSE.append(nrmse_.detach().cpu().numpy())
            pbar.set_description(f"train_loss: {train_loss:.3e}, NRMSE: {nrmse_:.4f}")
            pbar.update(epoch_plot_step)
    pbar.close()

    print("Training completed.")
    print(f"Final training loss: {losses[-1]:.3e}, NRMSE: {NRMSE[-1]:.4f}")
    # Save the model
    save_path = Path(mat_path)  # ← convert str to Path
    model_save_path = save_path.parent / "cvae_model.pth"
    torch.save(model.state_dict(), model_save_path)

    latent_space = model.reparameterize(mu, logvar)
    latent_space = latent_space.to("cpu").detach().numpy()

    recon_eddy_current = recon_x.to("cpu").detach().numpy()
    recon_wo_eddy_current = (
        model.decode(mu, InferenceCondition).to("cpu").detach().numpy()
    )

    return str(model_save_path), latent_space, recon_eddy_current, recon_wo_eddy_current


def inference(mat_path, model_path, motion_vector, inference_condition):
    # Load the .mat file (saved with -v7.3 in MATLAB)
    data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    ExtractToPython = data["ExtractToPython"]

    # Extract variables (examples below, adapt as needed)
    Phi_rt_init = (
        torch.from_numpy(ExtractToPython.Phi_rt_init.transpose()).float().to(device)
    )
    realtime_inference_condition = (
        torch.from_numpy(ExtractToPython.InferenceCondition).float().to(device)
    )
    md_inference_condition = torch.from_numpy(inference_condition).float().to(device)
    motion_vector = torch.from_numpy(motion_vector).float().to(device)

    feature_size = Phi_rt_init.size()[1]
    latent_dim = ExtractToPython.NumofLatent
    class_size = ExtractToPython.NumofCon
    sampling_rate = ExtractToPython.sampling_rate
    LPF = ExtractToPython.LPF
    BPFlb = ExtractToPython.BPFlb
    BPFub = ExtractToPython.BPFub

    model = CVAE_BP_HT(
        feature_size, latent_dim, class_size, sampling_rate, LPF, BPFlb, BPFub
    ).to(device)

    # Load the trained model parameters

    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        realtime, mu, logvar = model.decode(Phi_rt_init, realtime_inference_condition)
        recon_x, mu, logvar = model.decode(motion_vector, md_inference_condition)

    Phi_gen = recon_x.to("cpu").detach().numpy()

    return realtime, Phi_gen


model_path, latent_space, recon_eddy_current, recon_wo_eddy_current = train(input_mat)

# Save results back to MATLAB
scipy.io.savemat(
    output_mat,
    {
        "latent_space": latent_space,
        "model_path": model_path,
        "recon_eddy_current": recon_eddy_current,
        "recon_wo_eddy_current": recon_wo_eddy_current,
    },
)
