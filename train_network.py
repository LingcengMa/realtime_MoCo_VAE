import os
import sys
import torch

# torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
from pathlib import Path
from tqdm import tqdm
from network import CVAE_BP_HT
import numpy as np
import scipy
import matplotlib.pyplot as plt
import mat73

plt.rcParams["figure.dpi"] = 200
torch.autograd.set_detect_anomaly(True)

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# getting input and output paths from matlab
input_mat = sys.argv[1]
output_mat = sys.argv[2]

# input_mat = "/mnt/local_raid/xfang/Multitasking_SMS/Human_SMS_20210618_1/2D_SMS_DATA/meas_MID00063_FID30062_2D_Multitasking_Multiband_2FA_HYPT2Prep+IR_run1_20250910T112636/ExtractToPython.mat"
# output_mat = "/mnt/local_raid/xfang/Multitasking_SMS/Human_SMS_20210618_1/2D_SMS_DATA/meas_MID00063_FID30062_2D_Multitasking_Multiband_2FA_HYPT2Prep+IR_run1_20250910T112636/Output.mat"


# Define the loss function for the``                                         CVAE
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


def reset_weights(m, seed):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(
            m.weight, 0, "fan_in", "leaky_relu", torch.cuda.manual_seed_all(seed)
        )
        # seed(8) NRMSE at lowest: 0.1974
        # torch.nn.init.kaiming_uniform_(m.weight,0,"fan_in","leaky_relu")
        if m.bias is not None:
            m.bias.data.fill_(0.01)


# def reset_weights(m, seed):
#     torch.manual_seed(seed)
#     if isinstance(m, nn.Linear):
#         torch.nn.init.kaiming_uniform_(
#             m.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu"
#         )
#         if m.bias is not None:
#             m.bias.data.fill_(0.01)


### training functions ###


def train(mat_path):
    print(f"Loading data from: {mat_path}")

    # Load the .mat file (saved with -v7.3 in MATLAB)
    data = mat73.loadmat(mat_path)

    # Access the top-level structure from MATLAB
    ExtractToPython = data["ExtractToPython"]

    Phi_rt_small = ExtractToPython["Phi_rt_small"]
    Phi_rt_small = Phi_rt_small / Phi_rt_small[:, 0].std()  # Normalize whole
    # Phi_rt_small = (Phi_rt_small - Phi_rt_small.mean(axis=1, keepdims=True)) / Phi_rt_small.std(axis=1, keepdims=True) #Normalize each feature
    Phi_rt_small_real = Phi_rt_small.real
    Phi_rt_small_imag = Phi_rt_small.imag
    Phi_rt_small_con = np.concatenate((Phi_rt_small_real, Phi_rt_small_imag))
    Phi_rt_small_con = Phi_rt_small_con.transpose()
    Phi_rt_small = torch.from_numpy(Phi_rt_small_con).float().to(device)
    feature_size = Phi_rt_small.shape[1]
    # Extract variables (examples below, adapt as needed)
    # Phi_rt_com = torch.from_numpy(ExtractToPython.Phi_rt_com).float().to(device)
    TrainingCondition = (
        torch.from_numpy(ExtractToPython["TrainingCondition"]).float().to(device)
    )
    InferenceCondition = (
        torch.from_numpy(ExtractToPython["InferenceCondition"]).float().to(device)
    )

    lr = float(ExtractToPython["lr"])
    NumofEpochs = int(ExtractToPython["NumofEpochs"])
    class_size = int(ExtractToPython["NumofCon"])
    KLD_index = float(ExtractToPython["KLDindex"])
    LPF = float(ExtractToPython["LPF"])
    BPFlb = float(ExtractToPython["BPFlb"])
    BPFub = float(ExtractToPython["BPFub"])
    sampling_rate = float(ExtractToPython["sampling_rate"])
    latent_dim = int(ExtractToPython["NumofLatent"])
    seed = int(ExtractToPython["seed"])

    print("Training setup:")
    print(f"  Phi_rt_small shape: {Phi_rt_small.shape}")
    print(f"  TrainingCondition shape: {TrainingCondition.shape}")
    print(f"  Learning Rate: {lr}, Epochs: {NumofEpochs}")
    print(f"  Class Size: {class_size}, KLD Index: {KLD_index}")
    print(
        f"  LPF: {LPF}, BPFlb: {BPFlb}, BPFub: {BPFub}, Sampling Rate: {sampling_rate}"
    )
    print(f"  Feature Size: {feature_size}, Latent Dim: {latent_dim}, Seed: {seed}")

    model = CVAE_BP_HT(
        feature_size, latent_dim, class_size, sampling_rate, LPF, BPFlb, BPFub
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # reset_weights(model, seed)  # Initialize weights with seed

    epoch_plot_step = 1000  # Plot every 1000 epochs

    losses = []
    NRMSE = []
    total = (Phi_rt_small**2).sum()
    pbar = tqdm(total=NumofEpochs, desc="Initializing")
    for epoch in range(NumofEpochs):
        recon_x, mu, logvar = model(Phi_rt_small, TrainingCondition)
        optimizer.zero_grad()
        loss = loss_function_KLD(recon_x, Phi_rt_small, mu, logvar, KLD_index)
        loss.backward()
        optimizer.step()
        if (epoch % epoch_plot_step) == (epoch_plot_step - 1):
            train_loss = loss.detach().cpu().numpy() / Phi_rt_small.shape[0]
            losses.append(train_loss)
            mu, logvar = model.encode(Phi_rt_small, TrainingCondition)
            recon_x = model.decode(mu, TrainingCondition)  # NO REPARAMETERIZATION
            mse = ((recon_x - Phi_rt_small) ** 2).sum()
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

    with torch.no_grad():
        mu, logvar = model.encode(Phi_rt_small, TrainingCondition)
        recon_x = model.decode(mu, TrainingCondition)

    latent_space = mu.to("cpu").detach().numpy()
    recon_ec = model.decode(mu, InferenceCondition)

    recon_x = recon_x.to("cpu").detach().numpy()
    recon_ec = recon_ec.to("cpu").detach().numpy()

    return (
        str(model_save_path),
        latent_space,
        recon_x,
        recon_ec,
        TrainingCondition.to("cpu").detach().numpy(),
        InferenceCondition.to("cpu").detach().numpy(),
    )


# Run the training function
model_path, latent_space, recon_x, recon_ec, TrainingCondition, InferenceCondition = (
    train(input_mat)
)

# Save results back to MATLAB
scipy.io.savemat(
    output_mat,
    {
        "latent_space": latent_space,
        "model_path": model_path,
        "recon_x": recon_x,
        "recon_ec": recon_ec,
    },
)

# # --- FFT of recon_x ---
# fft_recon = np.fft.fft(recon_x, axis=0)  # FFT along time (T=25920)
# fft_recon_mag = np.abs(fft_recon[: fft_recon.shape[0] // 2, :])  # (12960, N)
# fft_recon_mean = np.mean(fft_recon_mag, axis=1)  # (12960,)

# # --- FFT of recon_ec ---
# fft_recon_ec = np.fft.fft(recon_ec, axis=0)
# fft_recon_ec_mag = np.abs(fft_recon_ec[: fft_recon_ec.shape[0] // 2, :])
# fft_recon_ec_mean = np.mean(fft_recon_ec_mag, axis=1)

# # --- FFT of exp(1j * thetas * pi / 180) ---
# thetas = TrainingCondition[:, 3]  # shape (T,)
# exp_signal = np.exp(1j * thetas * np.pi / 180)
# exp_signal = exp_signal - np.mean(exp_signal)  # remove DC component
# fft_theta = np.fft.fft(exp_signal)
# fft_theta_mag = np.abs(np.fft.fftshift(fft_theta))


# # --- Frequency axis ---
# n_timepoints = recon_x.shape[0]  # T = 25920
# freqs = np.fft.fftfreq(n_timepoints, d=0.07)
# freqs_shifted = np.fft.fftshift(freqs)

# # --- Shifted FFTs ---
# fft_recon_shifted = np.fft.fftshift(np.fft.fft(recon_x, axis=0), axes=0)
# fft_recon_mag = np.abs(fft_recon_shifted)
# fft_recon_mean = np.mean(fft_recon_mag, axis=1)

# fft_recon_ec_shifted = np.fft.fftshift(np.fft.fft(recon_ec, axis=0), axes=0)
# fft_recon_ec_mag = np.abs(fft_recon_ec_shifted)
# fft_recon_ec_mean = np.mean(fft_recon_ec_mag, axis=1)

# fft_theta_shifted = np.fft.fftshift(np.fft.fft(exp_signal))
# fft_theta_mag = np.abs(fft_theta_shifted)

# # --- Plot 1: Recon Output ---
# plt.figure(figsize=(8, 4))
# plt.plot(freqs_shifted, fft_recon_mean, label="Recon Output")
# plt.plot(freqs_shifted, fft_theta_mag, "--", label="exp(iθ) Reference")
# plt.title("FFT of Recon Output")
# plt.xlabel("Frequency (Hz or a.u.)")
# plt.ylabel("Magnitude")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("fft_recon_output_with_theta.png")
# plt.show()

# # --- Plot 2: Recon Output with Eddy Current ---
# plt.figure(figsize=(8, 4))
# plt.plot(
#     freqs_shifted, fft_recon_ec_mean, label="Recon Output (Eddy Current Corrected)"
# )
# plt.plot(freqs_shifted, fft_theta_mag, "--", label="exp(iθ) Reference")
# plt.title("FFT of Recon Output (Eddy Current Corrected)")
# plt.xlabel("Frequency (Hz or a.u.)")
# plt.ylabel("Magnitude")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("fft_recon_output_ec_with_theta.png")
# plt.show()
