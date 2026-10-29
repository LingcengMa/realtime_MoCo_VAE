import sys
import torch
import numpy as np
import mat73
import os
# torch.manual_seed(0)
import torch.utils
import torch.distributions
from network import CVAE_BP_HT
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
def inference(mat_path, model_path, inference_path):
    # Load the .mat file (saved with -v7.3 in MATLAB)
    data = mat73.loadmat(mat_path)
    inference = mat73.loadmat(inference_path)
    ExtractToPython = data["ExtractToPython"]
    # Extract variables (examples below, adapt as needed)
    Phi_rt_small = ExtractToPython["Phi_rt_small"]
    Phi_rt_small = Phi_rt_small / Phi_rt_small[:, 0].std()  # Normalize whole
    # Phi_rt_small = (Phi_rt_small - Phi_rt_small.mean(axis=1, keepdims=True)) / Phi_rt_small.std(axis=1, keepdims=True) #Normalize each feature
    Phi_rt_small_real = Phi_rt_small.real
    Phi_rt_small_imag = Phi_rt_small.imag
    Phi_rt_small_con = np.concatenate((Phi_rt_small_real, Phi_rt_small_imag))
    Phi_rt_small_con = Phi_rt_small_con.transpose()
    Phi_rt_small = torch.from_numpy(Phi_rt_small_con).float().to(device)
    feature_size = Phi_rt_small.shape[1]
    motion_vector = torch.from_numpy(inference["motion_vector"]).float().to(device)
    inference_condition = (
        torch.from_numpy(inference["inference_condition"]).float().to(device)
    )
    if motion_vector.dim() == 1:
        motion_vector = motion_vector.reshape(-1, 1)
    if inference_condition.dim() == 1:
        inference_condition = inference_condition.reshape(-1, 1)
    class_size = int(ExtractToPython["NumofCon"])
    LPF = float(ExtractToPython["LPF"])
    BPFlb = float(ExtractToPython["BPFlb"])
    BPFub = float(ExtractToPython["BPFub"])
    sampling_rate = float(ExtractToPython["sampling_rate"])
    latent_dim = int(ExtractToPython["NumofLatent"])
    model = CVAE_BP_HT(
        feature_size, latent_dim, class_size, sampling_rate, LPF, BPFlb, BPFub
    ).to(device)
    # Load the trained model parameters
    model.load_state_dict(torch.load(model_path))
    with torch.no_grad():
        recon_x = model.decode(motion_vector, inference_condition)
    Phi_gen = recon_x.to("cpu").detach().numpy()
    return Phi_gen
Phi_gen = inference(input_mat, model_path, inference_path)
scipy.io.savemat(
    recon_output_path,
    {
        "Phi_gen": Phi_gen,
    },
)
