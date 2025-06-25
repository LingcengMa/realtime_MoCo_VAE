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
from network import CVAE_BP_HT
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

if False:  # to improve: do this with a flag
    data_file = os.path.join(data_folder, "realtime_GT_imgs.mat")
    mat = scipy.io.loadmat(data_file)
    Phi_rt_init = mat["Phi_rt_init"]
else:
    # data_file = '/home/xfang/Data/stemberg_donna/ExtractToPython.mat'
    data_file = "data/cine_GRE/ExtractToPython_1.mat"
    mat = mat73.loadmat(data_file)
    mat_data = mat["ExtractToPython"]
    Phi_rt_init = mat_data["Phi_rt_small"]
    Segidx = mat_data["Segidx"].reshape(-1, 1)
    wallClock = mat_data["wallClock"].reshape(-1, 1)
    # flipAngles = mat_data["flipAngle"].reshape(-1, 1)
    thetas = mat_data["thetas"].reshape(-1, 1)

    Phi_rt_init = Phi_rt_init / Phi_rt_init[:, 0].std()  # Normalize whole
    # Phi_rt_init = (Phi_rt_init - Phi_rt_init.mean(axis=1, keepdims=True)) / Phi_rt_init.std(axis=1, keepdims=True) #Normalize each feature
    Phi_rt_init_real = Phi_rt_init.real
    Phi_rt_init_imag = Phi_rt_init.imag
    Phi_rt_init_con = np.concatenate((Phi_rt_init_real, Phi_rt_init_imag))
    Phi_rt_init_con = Phi_rt_init_con.transpose()
    Phi_rt_init = torch.from_numpy(Phi_rt_init_con).float().to(device)
    thetas_mu = (thetas - thetas.mean()) / thetas.std()
    thetas_mu = torch.from_numpy(thetas_mu).to(device).to(torch.float32)
    theta0 = torch.zeros_like(thetas_mu)

feature_size = Phi_rt_init.size()[1]
model = CVAE_BP_HT(feature_size, latent_dims=2, class_size=1).to(device)
model.load_state_dict(torch.load("cine_models/model_ExtractToPython_1.pt"))

recon_output_ec, latent_space_mu, latent_space_logvar = model(Phi_rt_init, thetas_mu)
recon_output = model.decode(latent_space_mu, theta0).to("cpu").detach().numpy()
print(recon_output.shape)
latent_space_mu_card_full = latent_space_mu
latent_space = model.reparameterize(latent_space_mu, latent_space_logvar)
latent_space = latent_space.to("cpu").detach().numpy()
latent_space_mu = latent_space_mu.to("cpu").detach().numpy()
# np.save("./cine_data/latent_GRE.npy", latent_space_mu)
# savemat("/home/xfang/Data/cine_data/latent_GRE.mat", {"latent": latent_space_mu})

# plot the distribution of latent space
resp = latent_space_mu[:, 0]
resp_min = np.percentile(resp, 10)
resp_max = np.percentile(resp, 90)
counts, bins, patches = plt.hist(resp, bins=100, edgecolor="black")
resp0_index = np.argmax(counts)
bin_range = (bins[resp0_index], bins[resp0_index + 1])
resp0 = (bin_range[0] + bin_range[1]) / 2

cardiac1 = latent_space_mu[:,1]
cardiac2 = latent_space_mu[:,2]

card_amplitude = np.sqrt(cardiac1**2 + cardiac2**2)
average_amplitude = np.mean(card_amplitude)
max_amplitude = np.percentile(card_amplitude, 99)
min_amplitude = np.percentile(card_amplitude, 1)
card_phase = np.arctan2(cardiac2, cardiac1)

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
    phase00 = smoothed_phase[min_index - 4]  # Phase at the previous point
    unique_phase, unique_indices = np.unique(smoothed_phase, return_index=True)
    unique_amplitude = smoothed_amplitude[unique_indices]  # Keep only unique amplitude
    amplitude_function = interp1d(unique_phase, unique_amplitude, kind="cubic", fill_value="extrapolate")
    return phase0, phase00, amplitude_function

phase0, phase00, amplitude_function = average_cycle(cardiac1, cardiac2)

cbins = 50
rbins = 20
contrast_num = 360
conditional_input = 1
conditional = theta0[:contrast_num]


latent_space_mu_fitting = torch.ones(contrast_num*cbins*rbins, 3).to(device)
resp_bins = torch.linspace(resp_min, resp_max, rbins)
latent_space_mu_fitting[:,0] = resp_bins.repeat_interleave(contrast_num*cbins)
card_phase_values = phase00 * torch.ones(cbins,1).to(device)
card_phase_bins = card_phase_values.repeat_interleave(contrast_num).repeat(rbins)
card_amplitude_resize = card_amplitude[::20][:50]
card_amplitude_values = torch.from_numpy(card_amplitude_resize).to(device)
card_amplitude_bins = card_amplitude_values.repeat_interleave(contrast_num).repeat(rbins)
latent_space_mu_fitting[:,1] = card_amplitude_bins * torch.cos(card_phase_bins)
latent_space_mu_fitting[:,2] = card_amplitude_bins * torch.sin(card_phase_bins)
condition_fitting = conditional.repeat(cbins*rbins,1).to(device)
phi4fitting = model.decode(latent_space_mu_fitting,condition_fitting).to("cpu").detach().numpy()

recon_output_old = np.load("assets/ExtractToPython_GRE_recon_output.npy")

# fixed phase
num_card_phase = 100
latent_space_mu_phase = torch.ones(num_card_phase, 3).to(device)
latent_space_mu_phase[:, 0] = resp0 * latent_space_mu_phase[:, 0]
cardiac_motion = np.linspace(min_amplitude, max_amplitude, num_card_phase)
latent_space_mu_phase[:, 1] = torch.from_numpy(cardiac_motion * np.cos(phase0))
latent_space_mu_phase[:, 2] = torch.from_numpy(cardiac_motion * np.sin(phase0))
card_fixed_phase = (
    model.decode(latent_space_mu_phase, theta0[:num_card_phase, :])
    .to("cpu")
    .detach()
    .numpy()
)  # NO REPARAMETERIZATION

# average amplitude
latent_space_mu_amplitude = torch.ones(num_card_phase, 3).to(device)
latent_space_mu_amplitude[:, 0] = resp0 * latent_space_mu_amplitude[:, 0]
phase_values = np.linspace(phase0, phase0 + 2 * np.pi, num_card_phase)
latent_space_mu_amplitude[:, 1] = torch.from_numpy(average_amplitude * np.cos(phase_values))
latent_space_mu_amplitude[:, 2] = torch.from_numpy(average_amplitude * np.sin(phase_values))
card_fixed_amplitude = (
    model.decode(latent_space_mu_amplitude, theta0[:num_card_phase, :])
    .to("cpu")
    .detach()
    .numpy()
)  # NO REPARAMETERIZATION


# full cardiac cycle pausing respiratory
latent_space_mu_card_full[:, 0] = resp0 * torch.ones_like(
    latent_space_mu_card_full[:, 0]
)
recon_output_card_full = (
    model.decode(latent_space_mu_card_full, theta0).to("cpu").detach().numpy()
)  # NO REPARAMETERIZATION

savemat(
    "/home/xfang/Data/cine_data/python_GRE_bloodpool.mat",
    {
        "realtime": recon_output,
        "phi4fitting": phi4fitting,
        "latentspace": latent_space_mu,
        "endofdiastole": phase0,
        "card_phase": card_fixed_phase,
        "card_amplitude": card_fixed_amplitude,
        "card_full": recon_output_card_full,
    },
)




# # pause respiratory
# latent_space_mu_card = torch.ones(num_card_phase, 3).to(device)
# latent_space_mu_card[:, 0] = resp0 * latent_space_mu_card[:, 0]
# latent_space_mu_card[:, 1] = torch.from_numpy(amplitude * np.cos(card_phase))
# latent_space_mu_card[:, 2] = torch.from_numpy(amplitude * np.sin(card_phase))
# recon_output_card = (
#     model.decode(latent_space_mu_card, theta0[:num_card_phase, :])
#     .to("cpu")
#     .detach()
#     .numpy()
# )  # NO REPARAMETERIZATION
# latent_space_mu_card_full[:, 0] = resp0 * torch.ones_like(
#     latent_space_mu_card_full[:, 0]
# )
# recon_output_card_full = (
#     model.decode(latent_space_mu_card_full, theta0).to("cpu").detach().numpy()
# )  # NO REPARAMETERIZATION

# # pause cardiac
# latent_space_mu_resp = torch.ones(resp_phase, 3).to(device)
# latent_space_mu_resp[:, 1] = torch.from_numpy(card_interp(0) * np.ones(resp_phase))
# latent_space_mu_resp[:, 2] = torch.zeros(resp_phase)
# latent_space_mu_resp[:, 0] = torch.linspace(inspiration_end, expiration_end, resp_phase)
# recon_output_resp = (
#     model.decode(latent_space_mu_resp, theta0[:resp_phase, :])
#     .to("cpu")
#     .detach()
#     .numpy()
# )  # NO REPARAMETERIZATION

# savemat(
#     "/home/xfang/Data/cine_data/GRE.mat",
#     {
#         "resp": recon_output_resp,
#         "card": recon_output_card,
#         "realtime": recon_output,
#         "card_full": recon_output_card_full,
#     },
# )


### pause respiratory ###

if False:  # to improve: do this with a flag
    Phi_rt_init = Phi_rt_init / Phi_rt_init[:, 0].std()  # Normalize whole
    # Phi_rt_init = (Phi_rt_init - Phi_rt_init.mean(axis=1, keepdims=True)) / Phi_rt_init.std(axis=1, keepdims=True) #Normalize each feature
    Phi_rt_init_real = Phi_rt_init.real
    Phi_rt_init_imag = Phi_rt_init.imag
    Phi_rt_init_con = np.concatenate((Phi_rt_init_real, Phi_rt_init_imag))
    Phi_rt_init_con = Phi_rt_init_con.transpose()
    Phi_rt_init = torch.from_numpy(Phi_rt_init_con).float().to(device)

    Segidx_mu = (Segidx - Segidx.mean()) / Segidx.std()
    wallClock_mu = (wallClock - wallClock.mean()) / wallClock.std()
    thetas_mu = (thetas - thetas.mean()) / thetas.std()
    flipAngles_mu = (flipAngles - flipAngles.mean()) / flipAngles.std()
    wallClock_mu = torch.from_numpy(wallClock_mu).to(device)
    Segidx_mu = torch.from_numpy(Segidx_mu).to(device)
    thetas_mu = torch.from_numpy(thetas_mu).to(device)
    flipAngles_mu = torch.from_numpy(flipAngles_mu).to(device)
    theta0 = torch.zeros_like(thetas_mu)
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

    class_size = 4
    latent_dims = 2
    feature_size = Phi_rt_init.size()[1]

    model = CVAE_BP_HT(feature_size, latent_dims, class_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    model_folder = os.path.join(path, "models")
    data_folder = os.path.join(path, "data")
    output_folder = "./SMS_plots"

    # Check if the 'plots' folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(data_folder):

        data_path = os.path.join(data_folder, file_name)
        mat = mat73.loadmat(data_file)
        mat_data = mat["ExtractToPython"]
        Phi_rt_init = mat_data["Phi_rt_small"]
        Phi_rt_init = Phi_rt_init / Phi_rt_init[:, 0].std()  # Normalize whole
        # Phi_rt_init = (Phi_rt_init - Phi_rt_init.mean(axis=1, keepdims=True)) / Phi_rt_init.std(axis=1, keepdims=True) #Normalize each feature
        Phi_rt_init_real = Phi_rt_init.real
        Phi_rt_init_imag = Phi_rt_init.imag
        Phi_rt_init_con = np.concatenate((Phi_rt_init_real, Phi_rt_init_imag))
        Phi_rt_init_con = Phi_rt_init_con.transpose()
        Phi_rt_init = torch.from_numpy(Phi_rt_init_con).float().to(device)

        model_path = os.path.join(model_folder, f"model_{file_name}.pth")
        model.load_state_dict(torch.load(model_path))
        recon_output_ec, latent_space_mu, latent_space_logvar = model(
            Phi_rt_init, t1t2fatheta
        )
        recon_output_ec = recon_output_ec.to("cpu").detach().numpy()  # w/ eddy current
        recon_output = model.decode(
            latent_space_mu, t1t2fatheta0
        )  # NO REPARAMETERIZATION
        recon_output = recon_output.to("cpu").detach().numpy()
        latent_space = model.reparameterize(latent_space_mu, latent_space_logvar)
        latent_space = latent_space.to("cpu").detach().numpy()
        latent_space_mu = latent_space_mu.to("cpu").detach().numpy()

        # output plots
        fig1, axs1 = plt.subplots(4, 2, figsize=(10, 10))
        axs1 = axs1.flatten()  # Flatten the 2x2 array of axes for easy indexing
        for i in range(4):
            axs1[i].plot(Phi_rt_init[:3600, i].cpu(), label=f"Phi_rt_init Col {i+1}")
            axs1[i].plot(
                recon_output_ec[:3600, i],
                label=f"Recon_output w/eddy current Col {i+1}",
            )
            axs1[i].legend(loc="upper right")
        for i in range(4):
            axs1[i + 4].plot(
                Phi_rt_init[:3600, i].cpu(), label=f"Phi_rt_init Col {i+1}"
            )
            axs1[i + 4].plot(recon_output[:3600, i], label=f"Recon_output Col {i+1}")
            axs1[i + 4].legend(loc="upper right")
        plt.tight_layout()
        fig1.savefig(os.path.join(output_folder, f"{file_name}_recon_output.png"))
        # latent vectors
        fig2, axs2 = plt.subplots(2, 3, figsize=(10, 10))
        axs2 = axs2.flatten()  # Flatten the 2x2 array of axes for easy indexing
        for i in range(3):
            axs2[i].plot(latent_space_mu[:3600, i], label=f"Latent vector {i+1}")
            axs2[i].legend(loc="upper right")
        axs2[3].plot(Segidx[:3600], label="Segidx")
        axs2[3].legend(loc="upper right")
        axs2[4].plot(wallClock[:3600], label="T2prep index")
        axs2[4].legend(loc="upper right")
        axs2[5].plot(flipAngles[:3600], label="Flip Angles")
        axs2[5].legend(loc="upper right")
        plt.tight_layout()
        fig2.savefig(os.path.join(output_folder, f"{file_name}_latent_vector.png"))


# latent_space_mu, latent_space_logvar = model.encode(Phi_rt_init, t1t2fatheta)
# latent_space = model.reparameterize(latent_space_mu,latent_space_logvar)
# latent_space = latent_space.to('cpu').detach().numpy()

# recon_output = model.decode(latent_space_mu, t1t2fatheta0) #NO REPARAMETERIZATION
# recon_output = recon_output.to('cpu').detach().numpy()
# np.save('./SMS/recon_output_0528_run1.npy', recon_output)

# cardiac_1 = latent_space_mu[:,1]
# cardiac_2 = latent_space_mu[:,2]
# cardiac_1 = cardiac_1.to('cpu').detach().numpy()
# cardiac_2 = cardiac_2.to('cpu').detach().numpy()
# np.save('./SMS/cardiac_1.npy', cardiac_1)
# np.save('./SMS/cardiac_2.npy', cardiac_2)
# amplitude_average = torch.mean(torch.sqrt(latent_space_mu[:,1]**2 + latent_space_mu[:,2]**2))
# print(amplitude_average.item())

# # plot the distribution of latent space
# resp = latent_space[:,0]
# fig = plt.hist(resp, bins=100, edgecolor='black')
# plt.title('Respiratory Distribution')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.savefig('./SMS/resp_distribution_0528_run1.png')

# ### pause contrast ###
# ### segidx at frame 120 ###

# Segidx_value = Segidx_mu[117,:] #
# Segidx_fixed = Segidx_mu
# Segidx_fixed = Segidx_fixed.fill_(Segidx_value.item())

# wallClock_value = torch.min(wallClock_mu)
# wallClock_fixed = wallClock_mu
# wallClock_fixed = wallClock_fixed.fill_(wallClock_value.item())

# T1T2fixedtheta0 = torch.cat([Segidx_fixed, wallClock_fixed, theta0], dim=1).to(torch.float32)
# t1t2fixedfatheta0 = torch.cat([Segidx_fixed, wallClock_fixed, flipAngles_mu, theta0], dim=1).to(torch.float32)

# ### pause respiratory ###
# ### resp end of expiration ###

# # expiration_end = -1.2
# # inspiration_end = 1.2

# # expiration_end = 1.3 # for mid
# # inspiration_end = -1.3 # for mid
# expiration_end = 0.75 # for cv
# inspiration_end = -0.4 # for cv
# resp_phase = 20

# ### pause cardiac ###
# ### cardiac at 153 ###

# cardiac_phase = 50
# diastole_end1 = latent_space_mu[139,1] # for cv
# diastole_end2 = latent_space_mu[139,2] # for cv
# # diastole_end1 = latent_space_mu[135,1] # for mid
# # diastole_end2 = latent_space_mu[135,2] # for mid
# # diastole_end1 = latent_space_mu[152,1]
# # diastole_end2 = latent_space_mu[152,2]
# # calculate the mean of the square root of (latent_space_mu[:,1]^2 + latent_space_mu[:,2]^2)
# amplitude_average = torch.mean(torch.sqrt(latent_space_mu[:,1]**2 + latent_space_mu[:,2]**2))

# latent_space = model.reparameterize(latent_space_mu,latent_space_logvar)
# latent_space = latent_space.to('cpu').detach().numpy()
# cardiac_1 = latent_space_mu[:,1]
# cardiac_2 = latent_space_mu[:,2]
# cardiac_1 = cardiac_1.to('cpu').detach().numpy()
# cardiac_2 = cardiac_2.to('cpu').detach().numpy()
# np.save('./SMS/0528_run1_cardiac_1.npy', cardiac_1)
# np.save('./SMS/0528_run1_cardiac_2.npy', cardiac_2)
# u1_fft = torch.fft.fft(latent_space_mu[:,0]).to('cpu').detach().numpy()
# u1_freq = torch.fft.fftfreq(len(latent_space_mu[:,0]), d=0.00712).to('cpu').detach().numpy()
# u2_fft = torch.fft.fft(latent_space_mu[:,1]).to('cpu').detach().numpy()
# u2_freq = torch.fft.fftfreq(len(latent_space_mu[:,1]), d=0.00712).to('cpu').detach().numpy()
# u3_fft = torch.fft.fft(latent_space_mu[:,2]).to('cpu').detach().numpy()
# u3_freq = torch.fft.fftfreq(len(latent_space_mu[:,2]), d=0.00712).to('cpu').detach().numpy()
# np.save('./SMS/0528_run1_latent_space.npy', latent_space)

# ### contrast: paused; respiratory: paused at end of expiration; cardiac: sinsoidal ###
# latent_space_mu_sin_card = torch.ones(cardiac_phase, latent_dims+1).to(device)
# latent_space_mu_sin_card[:,0] = expiration_end * latent_space_mu_sin_card[:,0]
# latent_space_mu_sin_card[:,1] = amplitude_average * torch.cos(torch.linspace(0, 2 * torch.pi, cardiac_phase)).to(device)
# latent_space_mu_sin_card[:,2] = amplitude_average * torch.sin(torch.linspace(0, 2 * torch.pi, cardiac_phase)).to(device)
# recon_output = model.decode(latent_space_mu_sin_card, t1t2fixedfatheta0[:cardiac_phase, :]) #NO REPARAMETERIZATION
# recon_output = recon_output.to('cpu').detach().numpy()
# np.save('./SMS/recon_output_card_0528_run1.npy', recon_output)
# # latent_space_mu_sin_card = model.reparameterize(latent_space_mu_sin_card,latent_space_logvar[:cardiac_phase, :])
# # latent_space_mu_sin_card = latent_space_mu_sin_card.to('cpu').detach().numpy()
# # np.save('./SMS/latent_space_card_0528_run1.npy', latent_space_mu_sin_card)

# ### contrast: T1(t), T2(t); respiratory: paused at end of expiration; cardiac: paused at end of diastole ###
# total_t = 308
# latent_space_mu_contrast = torch.ones(total_t, latent_dims+1).to(device)
# latent_space_mu_contrast[:,0] = expiration_end * latent_space_mu_contrast[:,0]
# latent_space_mu_contrast[:,1] = diastole_end1 * latent_space_mu_contrast[:,1]
# latent_space_mu_contrast[:,2] = diastole_end2 * latent_space_mu_contrast[:,2]
# recon_output = model.decode(latent_space_mu_contrast, t1t2fatheta0[:total_t, :]) #NO REPARAMETERIZATION
# recon_output = recon_output.to('cpu').detach().numpy()
# np.save('./SMS/recon_output_contrast_0528_run1.npy', recon_output)
# # latent_space_mu_contrast = model.reparameterize(latent_space_mu_contrast,latent_space_logvar[:total_t, :])
# # latent_space_mu_contrast= latent_space_mu_contrast.to('cpu').detach().numpy()
# # # np.save('./SMS/latent_space_mu_contrast_run2.npy', latent_space_mu_sin_card)

# ### contrast: paused; respiratory: end of inspiration to end of expiration; cardiac: paused at end of diastole ###
# latent_space_mu_resp_phase = torch.ones(resp_phase, latent_dims+1).to(device)
# latent_space_mu_resp_phase[:,1] = diastole_end1 * latent_space_mu_resp_phase[:,1]
# latent_space_mu_resp_phase[:,2] = diastole_end2 * latent_space_mu_resp_phase[:,2]
# latent_space_mu_resp_phase[:,0] = torch.linspace(inspiration_end, expiration_end, resp_phase).to(device)
# recon_output = model.decode(latent_space_mu_resp_phase, t1t2fixedfatheta0[:resp_phase, :]) #NO REPARAMETERIZATION
# recon_output = recon_output.to('cpu').detach().numpy()
# np.save('./SMS/recon_output_resp_0528_run1.npy', recon_output)
# # latent_space_mu_resp_phase = model.reparameterize(latent_space_mu_resp_phase,latent_space_logvar[:resp_phase, :])
# # latent_space_mu_resp_phase = latent_space_mu_resp_phase.to('cpu').detach().numpy()
# # np.save('./SMS/latent_space_mu_resp_0528_run1.npy', latent_space_mu_resp_phase)
