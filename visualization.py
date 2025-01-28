import os
import torch

torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions

# from models import VariationalAutoencoder_hc, VariationalAutoencoder_hc_m
import torchvision
import numpy as np
import scipy
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200
from network import CVAE_BP_HT
from PIL import Image
import cv2
import imageio.v2 as imageio
import scipy.stats as stats
import statsmodels.api as sm
from scipy.interpolate import CubicSpline

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# video visualization
latent_vector_GRE = np.load("./assets/ExtractToPython_GRE_latent_space.npy")
latent_vector_FISP = np.load("./assets/ExtractToPython_TrueFisp_latent_space.npy")

plt.figure(figsize=(9, 9))
plt.grid(True)
plt.subplot(2, 1, 1)
plt.plot(latent_vector_GRE[:100, 1])
plt.title("Cardiac 1")
plt.subplot(2, 1, 2)
plt.plot(latent_vector_GRE[:100, 2])
plt.title("Cardiac 2")
plt.savefig("./assets/latent_space_plot_GRE.png")

# # cardiac circle
# cardiac1 = latent_vector_GRE[:, 1]
# # cardiac1 = cardiac1[:, np.newaxis]
# cardiac2 = latent_vector_GRE[:, 2]
# # cardiac2 = cardiac2[:, np.newaxis]
# card_amplitude = np.sqrt(cardiac1**2 + cardiac2**2)
# card_phase = np.arctan2(cardiac2, cardiac1)
# # Mirror phase array to [-2π, 2π]
# phase_left = card_phase - 2 * np.pi  # Shift by -2π
# phase_right = card_phase + 2 * np.pi  # Shift by +2π
# # Concatenate arrays for extended phase and amplitude
# extended_card_phase = np.concatenate([phase_left, card_phase, phase_right])
# extended_card_amplitude = np.concatenate([card_amplitude, card_amplitude, card_amplitude])
# card_average = sm.nonparametric.lowess(extended_card_amplitude, extended_card_phase, frac= 1./3)
# np.save('./assets/card_average_GRE.npy', card_average)
# #scatter plot of the latent space
# card_interp = CubicSpline(card_average[:,0], card_average[:,1])
# phase = np.linspace(-np.pi, np.pi, 200)
# amplitude = card_interp(phase)
# print(card_interp(-np.pi))
# print(card_interp(np.pi))
# plt.figure(figsize=(9, 9))
# plt.plot(amplitude*np.cos(phase), amplitude*np.sin(phase))
# plt.savefig('./assets/latent_space_GRE.png')
# cardiac_trajectory = []


if False:

    for i in range(150, 300):
        plt.figure(figsize=(9, 9))

        # Plot the trajectory so far
        plt.plot(
            cardiac1[150 : i + 1], cardiac2[150 : i + 1], "b-", label="Trajectory"
        )  # Trajectory in blue

        # Plot the current position
        plt.plot(
            cardiac1[i], cardiac2[i], "ro", label="Current Position"
        )  # Current position in red

        # Set limits and labels
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.xlabel("X")
        plt.ylabel("Y")

        # Place axes at (0, 0)
        ax = plt.gca()
        ax.spines["left"].set_position("zero")
        ax.spines["bottom"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

        # Show grid for better visualization (optional)
        plt.grid(True)

        # Add legend
        plt.legend()

        # Save the plot as an image in the list
        plt.savefig(f"plot_{i}.png")
        cardiac_trajectory.append(imageio.imread(f"plot_{i}.png"))

        plt.close()

    # Create the GIF
    imageio.mimsave(
        "./assets/cardiac_trajectory_GRE.gif", cardiac_trajectory, duration=0.001
    )
    plt.figure(figsize=(9, 9))

    # Plot the trajectory so far
    plt.plot(
        cardiac1[150:300], cardiac2[150:300], "b-", label="Trajectory"
    )  # Trajectory in blue

    # Generate points for the circle
    x = amplitude * np.cos(phase)
    y = amplitude * np.sin(phase)

    # Plot the circle
    plt.plot(x, y, label="average cardiac cycle", color="red", linewidth=4)

    # Set equal scaling so the circle isn't skewed
    plt.gca().set_aspect("equal", adjustable="box")

    # Set limits and labels
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xlabel("Cardiac 1")
    plt.ylabel("Cardiac 2")

    # Place axes at (0, 0)
    ax = plt.gca()
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    # Show grid for better visualization (optional)
    plt.grid(True)

    # Add legend
    plt.legend()

    # Save the plot as an image in the list
    plt.savefig("plot_final.png")

    for i in range(150, 300):
        os.remove(f"plot_{i}.png")


# # cardiac circle
# cardiac1 = latent_vector_FISP[:, 1]
# # cardiac1 = cardiac1[:, np.newaxis]
# cardiac2 = latent_vector_FISP[:, 2]
# # cardiac2 = cardiac2[:, np.newaxis]
# card_amplitude = np.sqrt(cardiac1**2 + cardiac2**2)
# card_phase = np.arctan2(cardiac2, cardiac1)
# card_average = sm.nonparametric.lowess(card_amplitude, card_phase, frac= 1./3)
# #scatter plot of the latent space
# card_interp = CubicSpline(card_average[:,0], card_average[:,1])
# phase = np.linspace(-np.pi, np.pi, 200)
# amplitude = card_interp(phase)
# plt.figure(figsize=(9, 9))
# plt.plot(amplitude*np.cos(phase), amplitude*np.sin(phase))
# plt.savefig('./assets/latent_space_TrueFisp.png')

# cardiac_trajectory = []

if False:

    for i in range(0, 988):
        plt.figure(figsize=(9, 9))

        # Plot the trajectory so far
        plt.plot(
            cardiac1[0 : i + 1], cardiac2[0 : i + 1], "b-", label="Trajectory"
        )  # Trajectory in blue

        # Plot the current position
        plt.plot(
            cardiac1[i], cardiac2[i], "ro", label="Current Position"
        )  # Current position in red

        # Set limits and labels
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xlabel("X")
        plt.ylabel("Y")

        # Place axes at (0, 0)
        ax = plt.gca()
        ax.spines["left"].set_position("zero")
        ax.spines["bottom"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

        # Show grid for better visualization (optional)
        plt.grid(True)

        # Add legend
        plt.legend()

        # Save the plot as an image in the list
        plt.savefig(f"plot_{i}.png")
        cardiac_trajectory.append(imageio.imread(f"plot_{i}.png"))

        plt.close()

    # Create the GIF
    imageio.mimsave(
        "./assets/cardiac_trajectory_FISP.gif", cardiac_trajectory, duration=0.001
    )
    plt.figure(figsize=(9, 9))


# # Plot the trajectory so far
# plt.plot(cardiac1[90:1800], cardiac2[90:1800], 'b-', label='Trajectory')  # Trajectory in blue

# # Plot the current position
# plt.plot(cardiac1[1800], cardiac2[1800], 'ro', label='Current Position')  # Current position in red

# radius = 4.643391

# # Generate points for the circle
# theta = np.linspace(0, 2 * np.pi, 100)  # 100 points from 0 to 2π
# x = radius * np.cos(theta)
# y = radius * np.sin(theta)

# # Plot the circle
# plt.plot(x, y, label='average cardiac cycle', color='red', linewidth=4)

# # Set equal scaling so the circle isn't skewed
# plt.gca().set_aspect('equal', adjustable='box')

# # Set limits and labels
# plt.xlim(-7, 7)
# plt.ylim(-7, 7)
# plt.xlabel('X')
# plt.ylabel('Y')

# # Place axes at (0, 0)
# ax = plt.gca()
# ax.spines['left'].set_position('zero')
# ax.spines['bottom'].set_position('zero')
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')

# # Show grid for better visualization (optional)
# plt.grid(True)

# # Add legend
# plt.legend()

# # Save the plot as an image in the list
# plt.savefig('plot_1800.png')

# for i in range(0,988):
#     os.remove(f'plot_{i}.png')

# Load the data
# patient data
# original_rt_file = '/home/xfang/Data/gifs_qual/patient/recon_realtime_original.gif'
# recon_rt_file = '/home/xfang/Data/gifs_qual/patient/recon_realtime_wo_ec.gif'
# mt_card_file = '/home/xfang/Data/gifs_qual/patient/pat_mt_reconCard.gif'
# mt_resp_file = '/home/xfang/Data/gifs_qual/patient/pat_mt_reconResp.gif'
# mt_recovery_file = '/home/xfang/Data/gifs_qual/patient/pat_mt_reconRecovery.gif'
# mt_card_file_wavelet = '/home/xfang/Data/gifs_qual/patient/pat_mt_reconCard_wavelet.gif'
# mt_resp_file_wavelet = '/home/xfang/Data/gifs_qual/patient/pat_mt_reconResp_wavelet.gif'
# mt_recovery_file_wavelet = '/home/xfang/Data/gifs_qual/patient/pat_mt_reconRecovery_wavelet.gif'
# gen_card_file = '/home/xfang/Data/gifs_qual/patient/gen_cardiac.gif'
# gen_resp_file = '/home/xfang/Data/gifs_qual/patient/gen_respphase.gif'
# gen_recovery_file = '/home/xfang/Data/gifs_qual/patient/gen_reconrecovery.gif'

# # Path to the static images
# mt_card_path = '/home/xfang/projects/vae_temporal_analysis/recon_img/mt_card_d.png'
# mt_resp_path = '/home/xfang/projects/vae_temporal_analysis/recon_img/mt_resp_d.png'
# mt_recovery_path = '/home/xfang/projects/vae_temporal_analysis/recon_img/mt_recovery_d.png'
# mt_card_wavelet_path = '/home/xfang/projects/vae_temporal_analysis/recon_img/mt_card_wavelet_d.png'
# mt_resp_wavelet_path = '/home/xfang/projects/vae_temporal_analysis/recon_img/mt_resp_wavelet_d.png'
# mt_recovery_wavelet_path = '/home/xfang/projects/vae_temporal_analysis/recon_img/mt_recovery_wavelet_d.png'
# gen_card_path = '/home/xfang/projects/vae_temporal_analysis/recon_img/gen_card_d.png'
# gen_resp_path = '/home/xfang/projects/vae_temporal_analysis/recon_img/gen_resp_d.png'
# gen_recovery_path = '/home/xfang/projects/vae_temporal_analysis/recon_img/gen_recovery_d.png'
# original_rt_path = '/home/xfang/projects/vae_temporal_analysis/recon_img/original_d.png'
# recon_rt_path = '/home/xfang/projects/vae_temporal_analysis/recon_img/recon_rt_d.png'

# # # CV data
# original_rt_file = '/home/xfang/Data/gifs_qual/CV/cv_reconRealtime.gif'
# recon_rt_file = '/home/xfang/Data/gifs_qual/CV/cv_reconRealtime_HT_wo_eddy.gif'
# mt_card_file = '/home/xfang/Data/gifs_qual/CV/cv_reconCard.gif'
# mt_resp_file = '/home/xfang/Data/gifs_qual/CV/cv_reconResp.gif'
# mt_recovery_file = '/home/xfang/Data/gifs_qual/CV/cv_reconRecovery.gif'
# mt_card_file_wavelet = '/home/xfang/Data/gifs_qual/CV/cv_reconCard_wavelet.gif'
# mt_resp_file_wavelet = '/home/xfang/Data/gifs_qual/CV/cv_reconResp_wavelet.gif'
# mt_recovery_file_wavelet = '/home/xfang/Data/gifs_qual/CV/cv_reconRecovery_wavelet.gif'
# gen_card_file = '/home/xfang/Data/gifs_qual/CV/cv_generative_card.gif'
# gen_resp_file = '/home/xfang/Data/gifs_qual/CV/cv_generative_resp.gif'
# gen_recovery_file = '/home/xfang/Data/gifs_qual/CV/cv_generative_contrast.gif'

# # Path to the static images
# mt_card_path = '/home/xfang/projects/vae_temporal_analysis/CV_IMGS/mt_card_cv.png'
# mt_resp_path = '/home/xfang/projects/vae_temporal_analysis/CV_IMGS/mt_resp_cv.png'
# mt_recovery_path = '/home/xfang/projects/vae_temporal_analysis/CV_IMGS/mt_recovery_cv.png'
# mt_card_wavelet_path = '/home/xfang/projects/vae_temporal_analysis/CV_IMGS/mt_card_wavelet_cv.png'
# mt_resp_wavelet_path = '/home/xfang/projects/vae_temporal_analysis/CV_IMGS/mt_resp_wavelet_cv.png'
# mt_recovery_wavelet_path = '/home/xfang/projects/vae_temporal_analysis/CV_IMGS/mt_recovery_wavelet_cv.png'
# gen_card_path = '/home/xfang/projects/vae_temporal_analysis/CV_IMGS/gen_card_cv.png'
# gen_resp_path = '/home/xfang/projects/vae_temporal_analysis/CV_IMGS/gen_resp_cv.png'
# gen_recovery_path = '/home/xfang/projects/vae_temporal_analysis/CV_IMGS/gen_recovery_cv.png'
# original_rt_path = '/home/xfang/projects/vae_temporal_analysis/CV_IMGS/original_cv.png'
# recon_rt_path = '/home/xfang/projects/vae_temporal_analysis/CV_IMGS/recon_rt_cv.png'

# # # MID data
# original_rt_file = '/home/xfang/Data/gifs_qual/MID/mid_reconRealtime.gif'
# recon_rt_file = '/home/xfang/Data/gifs_qual/MID/mid_reconRealtime_HT_wo_eddy.gif'
# mt_card_file = '/home/xfang/Data/gifs_qual/MID/mid_reconCard.gif'
# mt_resp_file = '/home/xfang/Data/gifs_qual/MID/mid_reconResp.gif'
# mt_recovery_file = '/home/xfang/Data/gifs_qual/MID/mid_reconRecovery.gif'
# mt_card_file_wavelet = '/home/xfang/Data/gifs_qual/MID/mid_reconCard.gif'
# mt_resp_file_wavelet = '/home/xfang/Data/gifs_qual/MID/mid_reconResp.gif'
# mt_recovery_file_wavelet = '/home/xfang/Data/gifs_qual/MID/mid_reconRecovery.gif'
# gen_card_file = '/home/xfang/Data/gifs_qual/MID/mid_generative_card.gif'
# gen_resp_file = '/home/xfang/Data/gifs_qual/MID/mid_generative_resp.gif'
# gen_recovery_file = '/home/xfang/Data/gifs_qual/MID/mid_generative_contrast.gif'

# # Path to the static images
# mt_card_path = '/home/xfang/Data/gifs_qual/MID_IMGS/mt_card_cv.png'
# mt_resp_path = '/home/xfang/Data/gifs_qual/MID_IMGS/mt_resp_cv.png'
# mt_recovery_path = '/home/xfang/Data/gifs_qual/MID_IMGS/mt_recovery_cv.png'
# mt_card_wavelet_path = '/home/xfang/Data/gifs_qual/MID_IMGS/mt_card_wavelet_cv.png'
# mt_resp_wavelet_path = '/home/xfang/Data/gifs_qual/MID_IMGS/mt_resp_wavelet_cv.png'
# mt_recovery_wavelet_path = '/home/xfang/Data/gifs_qual/MID_IMGS/mt_recovery_wavelet_cv.png'
# gen_card_path = '/home/xfang/Data/gifs_qual/MID_IMGS/gen_card_cv.png'
# gen_resp_path = '/home/xfang/Data/gifs_qual/MID_IMGS/gen_resp_cv.png'
# gen_recovery_path = '/home/xfang/Data/gifs_qual/MID_IMGS/gen_recovery_cv.png'
# original_rt_path = '/home/xfang/Data/gifs_qual/MID_IMGS/original_cv.png'
# recon_rt_path = '/home/xfang/Data/gifs_qual/MID_IMGS/recon_rt_cv.png'


# # Open the GIF
# gif1 = Image.open(original_rt_file)
# gif2 = Image.open(recon_rt_file)

# gif3= Image.open(mt_card_file)
# gif4 = Image.open(mt_card_file_wavelet)
# gif5 = Image.open(gen_card_file)

# gif6= Image.open(mt_resp_file)
# gif7 = Image.open(mt_resp_file_wavelet)
# gif8 = Image.open(gen_resp_file)

# gif9= Image.open(mt_recovery_file)
# gif10 = Image.open(mt_recovery_file_wavelet)
# gif11 = Image.open(gen_recovery_file)

# # Create a list to store the rows

# for i in range(1,6):
#     rows = []
#     try:
#         while True:
#             if i == 1:
#                 gif = gif1
#             elif i == 2:
#                 gif = gif2
#             elif i == 3:
#                 gif = gif3
#             elif i == 4:
#                 gif = gif4
#             elif i == 5:
#                 gif = gif5
#             frame = gif.convert('RGB')
#             frame_array = np.array(frame)
#             row = frame_array[:,97, :] # 68 for cardiac and real-time, 97 for mid
#             rows.append(row)
#             gif.seek(gif.tell() + 1)
#             output_image_array = np.stack(rows, axis=1)
#             output_image = Image.fromarray(output_image_array)
#             if i == 1:
#                 output_image.save(original_rt_path)
#             elif i == 2:
#                 output_image.save(recon_rt_path)
#             elif i == 3:
#                 output_image.save(mt_card_path)
#             elif i == 4:
#                 output_image.save(mt_card_wavelet_path)
#             elif i == 5:
#                 output_image.save(gen_card_path)
#     except EOFError:
#         pass

# for i in range(6,9):
#     rows = []
#     try:
#         while True:
#             if i == 6:
#                 gif = gif6
#             elif i == 7:
#                 gif = gif7
#             elif i == 8:
#                 gif = gif8
#             frame = gif.convert('RGB')
#             frame_array = np.array(frame)
#             row = frame_array[145, :, :] # 120 for resp, 145 for mid resp
#             rows.append(row)
#             gif.seek(gif.tell() + 1)
#             output_image_array = np.stack(rows, axis=1)
#             output_image = Image.fromarray(output_image_array)
#             if i == 6:
#                 output_image.save(mt_resp_path)
#             elif i == 7:
#                 output_image.save(mt_resp_wavelet_path)
#             elif i == 8:
#                 output_image.save(gen_resp_path)
#     except EOFError:
#         pass

# for i in range(9,12):
#     rows = []
#     try:
#         while True:
#             if i == 9:
#                 gif = gif9
#             elif i == 10:
#                 gif = gif10
#             elif i == 11:
#                 gif = gif11
#             frame = gif.convert('RGB')
#             frame_array = np.array(frame)
#             row = frame_array[:, 97, :] # 75 for recovery
#             rows.append(row)
#             gif.seek(gif.tell() + 1)
#             output_image_array = np.stack(rows, axis=1)
#             output_image = Image.fromarray(output_image_array)
#             if i == 9:
#                 output_image.save(mt_recovery_path)
#             elif i == 10:
#                 output_image.save(mt_recovery_wavelet_path)
#             elif i == 11:
#                 output_image.save(gen_recovery_path)
#     except EOFError:
#         pass
