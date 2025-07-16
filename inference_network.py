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
model_path = sys.argv[2]
inference_path = sys.argv[3]
recon_output_path = sys.argv[4]
# input_mat = "/mnt/local_raid/xfang/Multitasking_SMS/Human_SMS_20210618_2/2D_SMS_DATA/meas_MID00159_FID30158_2D_Multitasking_Multiband_2FA_HYPT2Prep+IR_run1_20250715T232321/ExtractToPython_20210618_2_run1.mat"
# model_path = "/mnt/local_raid/xfang/Multitasking_SMS/Human_SMS_20210618_2/2D_SMS_DATA/meas_MID00159_FID30158_2D_Multitasking_Multiband_2FA_HYPT2Prep+IR_run1_20250715T232321/cvae_model.pth"
# inference_path = "/mnt/local_raid/xfang/Multitasking_SMS/Human_SMS_20210618_2/2D_SMS_DATA/meas_MID00159_FID30158_2D_Multitasking_Multiband_2FA_HYPT2Prep+IR_run1_20250715T232321/inference_inputs.mat"
# recon_output_path = "/mnt/local_raid/xfang/Multitasking_SMS/Human_SMS_20210618_2/2D_SMS_DATA/meas_MID00159_FID30158_2D_Multitasking_Multiband_2FA_HYPT2Prep+IR_run1_20250715T232321/recon_output.mat"


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


def inference(mat_path, model_path, inference_path):
    # Load the .mat file (saved with -v7.3 in MATLAB)
    data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    inference = scipy.io.loadmat(
        inference_path, struct_as_record=False, squeeze_me=True
    )

    ExtractToPython = data["ExtractToPython"]
    # Extract variables (examples below, adapt as needed)
    Phi_rt_init = (
        torch.from_numpy(ExtractToPython.Phi_rt_init.transpose()).float().to(device)
    )
    motion_vector = torch.from_numpy(inference["motion_vector"]).float().to(device)
    md_inference_condition = (
        torch.from_numpy(inference["md_inference_condition"]).float().to(device)
    )
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
        recon_x = model.decode(motion_vector, md_inference_condition)

    Phi_gen = recon_x.to("cpu").detach().numpy()

    return Phi_gen


Phi_gen = inference(input_mat, model_path, inference_path)

scipy.io.savemat(
    recon_output_path,
    {
        "Phi_gen": Phi_gen,
    },
)
