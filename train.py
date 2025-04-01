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
from torchvision import datasets
import numpy as np
import scipy
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200
torch.autograd.set_detect_anomaly(True)

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# data_init
path = os.path.dirname(os.path.realpath(__file__))


# functions
def loss_function_BP(recon_x, x, mu, mu_filtered, logvar):
    BCE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    FLT = torch.sum((mu - mu_filtered) ** 2)
    return BCE + 1e-9 * KLD + 1e-6 * FLT


def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + 1e-9 * KLD  # was 1e-9 for T1-T2 single-FA single-slice (prelim data)


def loss_function_KLD(recon_x, x, mu, logvar, kld_index):
    BCE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + kld_index * KLD


def low_pass_filter(x, upper_limit):
    x_fft = torch.fft.fft(x, dim=-1)
    x_freq = torch.fft.fftfreq(len(x), d=0.00712, device="cuda")  #
    filtered_x = torch.where(
        (abs(x_freq) <= upper_limit), x_fft, torch.zeros_like(x_fft)
    )
    x_filtered = torch.fft.ifft(filtered_x, dim=-1)
    return x_filtered


def band_pass_filter(x, upper_limit, lower_limit):
    x_fft = torch.fft.fft(x, dim=-1)
    x_freq = torch.fft.fftfreq(len(x), d=0.00712, device="cuda")
    filtered_x = torch.where(
        (abs(x_freq) >= lower_limit) & (abs(x_freq) <= upper_limit),
        x_fft,
        torch.zeros_like(x_fft),
    )
    x_filtered = torch.fft.ifft(filtered_x, dim=-1)
    return x_filtered


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


### evaluation mode ###
def train_model_kld(
    model,
    optimizer,
    max_epochs,
    epoch_plot_step,
    Phi_rt_init,
    conditional_input,
    kld_index,
):
    losses = []
    NRMSE = []
    total = (Phi_rt_init**2).sum()
    pbar = tqdm(total=max_epochs, desc="Initializing")
    for epoch in range(max_epochs):
        model.train()
        recon_x, mu, logvar = model(Phi_rt_init, conditional_input)
        optimizer.zero_grad()
        loss = loss_function_KLD(recon_x, Phi_rt_init, mu, logvar, kld_index)
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


def plot_latent_space(model, data, conditional, epoch):
    model.eval()
    with torch.no_grad():
        _, z, _ = model(data, conditional)

    z = z.to("cpu").detach().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    axs = axs.flatten()  # Flatten the 2x2 array of axes for easy indexing
    for i in range(3):
        axs[i].plot(z[:4000, i], label=f"Latent vector {i+1}")
        axs[i].legend(loc="upper right")
    plt.tight_layout()
    plt.title(f"Latent Space at Epoch {epoch}")
    fig.savefig(
        os.path.join(plot_folder, f"{file_name}_latent_vector at Epoch {epoch}.png")
    )


def train_long(
    model,
    optimizer,
    max_epochs,
    checkpoint,
    epoch_plot_step,
    Phi_rt_init,
    conditional_input,
):
    losses = []
    NRMSE = []
    total = (Phi_rt_init**2).sum()
    pbar = tqdm(total=max_epochs, desc="Initializing")
    for epoch in range(max_epochs):
        model.train()
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
        if (epoch % checkpoint) == 0:
            torch.save(
                model.state_dict(),
                os.path.join(model_folder, f"model_{file_name}_epoch_{epoch}.pth"),
            )
            plot_latent_space(model, Phi_rt_init, conditional_input, epoch)
    pbar.close()
    return losses, NRMSE


def evaluate_model(model, Phi_rt_init_val, conditional_input):
    model.eval()
    correct = 0
    total = (Phi_rt_init_val**2).sum()
    with torch.no_grad():
        (
            recon_output,
            _,
            _,
        ) = model(Phi_rt_init_val, conditional_input)
        mse = ((recon_output - Phi_rt_init_val) ** 2).sum()
        nrmse = torch.sqrt(mse / total)
    return nrmse


class DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z, c):
        return self.model.decode(z, c)  # Call the decode method


# if False: #to improve: do this with a flag
#     data_file = os.path.join(data_folder, 'realtime_GT_imgs.mat')
#     mat = scipy.io.loadmat(data_file)
#     Phi_rt_init = mat['Phi_rt_init']
#     seg_file = os.path.join(data_folder, 'TI.mat')
#     segidx = scipy.io.loadmat(seg_file)
#     segidx = segidx['Segidx']
#     binning_file = os.path.join(data_folder, 'binning_GT.mat')
#     binning_file = scipy.io.loadmat(binning_file)
#     hidx = binning_file['Hidx'] # Cardiac
#     ridx = binning_file['Ridx'] # Respiratory
#     # create simulated t2 prep signal
#     array_1 = np.ones(360)
#     array_2 = 2 * array_1
#     array_3 = 3 * array_1
#     array_4 = 4 * array_1
#     array_5 = 5 * array_1
#     array = np.concatenate((array_1, array_2, array_3),axis=0)
#     array = np.concatenate((array, array_4, array_5),axis=0)
#     array = np.tile(array, 7)

#     ridx_mu = (ridx - ridx.mean())/ridx.std()
#     hidx_mu = (hidx - hidx.mean())/hidx.std()
#     Segidx_mu = (segidx - segidx.mean())/segidx.std()
#     # ramp_latent_mu = (array-array.mean())/array.std()
#     wallClock_mu = (array - array.mean())/array.std()
#     # thetas_mu = (thetas - thetas.mean())/thetas.std()

# else :
#     # data_file = '/home/xfang/Data/Multitasking_SMS/Human_SMS_20210528/2D_SMS_DATA/ExtractToPython_2FA_run2.mat'
#     data_file = './data/ExtractToPython_0528_run1.mat'
#     # data_file = '/home/xfang/Data/stemberg_donna/ExtractToPython.mat'
#     mat = mat73.loadmat(data_file)
#     mat_data =mat['ExtractToPython']
#     Phi_rt_init = mat_data['Phi_rt_small']
#     Segidx = mat_data['Segidx'].reshape(-1, 1)
#     wallClock = mat_data['wallClock'].reshape(-1, 1)
#     thetas = mat_data['thetas'].reshape(-1, 1)

# hidx = mat['Hidx'] # Cardiac
# ridx = mat['Ridx'] # Respiratory
# thetas = mat['thetas'] # Thetas

if False:
    data_file = "./data/ExtractToPython_0528_run1.mat"
    mat = mat73.loadmat(data_file)
    mat_data = mat["ExtractToPython"]
    Phi_rt_init = mat_data["Phi_rt_small"]
    Segidx = mat_data["Segidx"].reshape(-1, 1)
    wallClock = mat_data["wallClock"].reshape(-1, 1)
    thetas = mat_data["thetas"].reshape(-1, 1)
    flipAngles = mat_data["flipAngle"].reshape(-1, 1)

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

# KLD indices to test
kld_indices = [
    1e-30,
    1e-27,
    1e-24,
    1e-21,
    1e-18,
    1e-15,
    1e-12,
    1e-9,
    1e-6,
    1e-3,
    1,
    1e3,
    1e6,
    1e9,
    1e12,
]
lr_grid = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]

data_folder = "./sms_data_test"
plot_folder = "./latent_plots"
model_folder = "./SMS_models"
assets_folder = "./SMS_assest"
# model_folder = './cine_models'
# assets_folder = './assets'


if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

if not os.path.exists(assets_folder):
    os.makedirs(assets_folder)

for file_name in os.listdir(data_folder):

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

    if True:
        #### training ####
        feature_size = Phi_rt_init.size()[1]
        model = CVAE_BP_HT(feature_size, latent_dims=2, class_size=4).to(device)
        model.apply(reset_weights)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        max_epochs = 60000
        epoch_plot_step = 10
        checkpoint = 10000

        print(f"Training on Data File: {file_name}")
        losses, NRMSE = train_model(
            model, optimizer, max_epochs, epoch_plot_step, Phi_rt_init, t1t2fatheta
        )
        # Save the model after training on the current dataset
        torch.save(
            model.state_dict(),
            os.path.join(model_folder, f"model_{file_name}.pth"),
        )
        print(f"Model saved as model_{file_name}.pth")
        # losses, NRMSE = train_long(model, optimizer, max_epochs, checkpoint, epoch_plot_step, Phi_rt_init, t1t2fatheta)
        # model = torch.load('SMS_models/model_ExtractToPython_0601_run1.pt')
        # example_c = torch.randn(1, 4).to(device)
        # example_z= torch.randn(1, 3).to(device)
        # Decoder = DecoderWrapper(model).to(device)

        # torch.onnx.export(
        #     Decoder,
        #     (example_z, example_c),
        #     '/home/xfang/Data/Multitasking_SMS/Human_SMS_20210601/2D_SMS_DATA/cvae_decoder.onnx',
        #     input_names=['latent_input', 'conditional_input'],
        #     output_names=['output'],
        #     dynamic_axes={
        #         'latent_input': {0: 'batch_size'},
        #         'conditional_input': {0: 'batch_size'},
        #         'output': {0: 'batch_size'}
        #     }
        # )

        # traced_model = torch.jit.trace_module(model.decode, (example_z, example_conditions))
        # torch.jit.save(traced_model, os.path.join(model_folder, f"traced_decoder_{file_name}.pt"),)
        # model.eval()

        with torch.no_grad():
            mu, logvar = model.encode(
                Phi_rt_init, t1t2fatheta
            )  # Get latent space representations
        # torch.save(mu, f"{file_name}_latent_vectors.pt")

        # mu = mu.to('cpu').detach().numpy()
        # scipy.io.savemat(f"/home/xfang/Data/Multitasking_SMS/Human_SMS_20210601/2D_SMS_DATA/{file_name}_latent_vectors.mat", {"latent": mu})

        # output = Decoder(mu, t1t2fatheta0).detach().cpu().numpy()
        # recon_output = model.decode(mu, t1t2fatheta0).detach().cpu().numpy()

        # fig2, axs2 = plt.subplots(2, 4, figsize=(20, 20))
        # axs2 = axs2.flatten()  # Flatten the 2x2 array of axes for easy indexing
        # for i in range(4):
        #     axs2[i].plot(recon_output[:3600, i], label = f'Phi_rt_init Col {i+1}')
        #     axs2[i].plot(output[:3600, i], label = f'Recon_output w/eddy current Col {i+1}')
        #     axs2[i].legend(loc="upper right")
        # for i in range(4):
        #     axs2[i+4].plot(recon_output[:3600, i], label=f'Phi_rt_init Col {i+1}')
        #     axs2[i+4].plot(output[:3600, i], label=f'Recon_output Col {i+1}')
        #     axs2[i+4].legend(loc="upper right")
        # plt.tight_layout()
        # fig2.savefig('testtest.png')

    else:
        #### training and validation ###
        total_size = Phi_rt_init.size(0)
        train_size = int(0.8 * total_size)  # 80%
        val_size = total_size - train_size  # 20%

        phi_train = Phi_rt_init[:train_size]  # First 80%
        phi_val = Phi_rt_init[train_size:]  # Last 20%

        conditional_train = t1t2fatheta[:train_size]
        conditional_val = t1t2fatheta[train_size:]

        # conditional_train = t1t2fatheta[:train_size]
        # conditional_val = t1t2fatheta[train_size:]

        feature_size = Phi_rt_init.size()[1]
        model = CVAE_BP_HT(feature_size, latent_dims=2, class_size=4).to(device)
        model.apply(reset_weights)

        # for lr_value in lr_grid:
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        # best lr = 0.0001 1e-4

        max_epochs = 40000
        epoch_plot_step = 10
        best_nrmse = 1
        train_nrmse_list = []
        val_nrmse_list = []

        for kld_index in kld_indices:
            model.apply(reset_weights)
            print(f"Training on Data File: {file_name} with KLD Index: {kld_index}")
            losses, train_nrmse = train_model_kld(
                model,
                optimizer,
                max_epochs,
                epoch_plot_step,
                phi_train,
                conditional_train,
                kld_index,
            )
            # Save the model after training on the current dataset
            torch.save(
                model.state_dict(),
                os.path.join(model_folder, f"model_{file_name}_kld_{kld_index}.pth"),
            )

            recon_output_ec, latent_space_mu, latent_space_logvar = model(
                Phi_rt_init, t1t2fatheta
            )
            recon_output_ec = (
                recon_output_ec.to("cpu").detach().numpy()
            )  # w/ eddy current
            recon_output = model.decode(
                latent_space_mu, t1t2fatheta0
            )  # NO REPARAMETERIZATION
            recon_output = recon_output.to("cpu").detach().numpy()
            latent_space_mu = latent_space_mu.to("cpu").detach().numpy()

            print(f"Model saved as model_{file_name}_kld_{kld_index}.pth")
            val_nrmse = evaluate_model(model, phi_val, conditional_val)
            train_nrmse_list.append(train_nrmse)
            val_nrmse_list.append(val_nrmse)

            fig, axs = plt.subplots(1, 3, figsize=(30, 10))
            # fig3, axs3 = plt.subplots(3, 1, figsize=(10, 10))
            axs = axs.flatten()  # Flatten the 2x2 array of axes for easy indexing
            for i in range(3):
                axs[i].plot(latent_space_mu[:3600, i], label=f"Latent vector {i+1}")
                axs[i].legend(loc="upper right")
            # axs[3].plot(Segidx[:3600], label='Segidx')
            # axs[3].legend(loc="upper right")
            # axs[4].plot(wallClock[:3600], label='T2prep index')
            # axs[4].legend(loc="upper right")
            # axs[5].plot(flipAngles[:3600], label='Flip Angles')
            # axs[5].legend(loc="upper right")
            # plot title
            plt.suptitle(f"KLD Index: {kld_index}")
            plt.tight_layout()
            fig.savefig(
                os.path.join(plot_folder, f"{file_name}_{kld_index}_latent.png")
            )

            print(f"Validation Accuracy: {val_nrmse}")
            if val_nrmse < best_nrmse:
                best_nrmse = val_nrmse
                best_kld_index = kld_index
                # best_lr = lr_value
                best_model = model

        # save train_nrmse_list and val_nrmse_list
        torch.save(
            train_nrmse_list,
            os.path.join(model_folder, f"train_nrmse_list_{file_name}.pth"),
        )
        torch.save(
            val_nrmse_list,
            os.path.join(model_folder, f"val_nrmse_list_{file_name}.pth"),
        )
        # Plot the NRMSE for different KLD indices

        plt.figure(figsize=(8, 5))
        plt.plot(
            kld_indices,
            train_nrmse_list.to("cpu").detach().numpy(),
            marker="o",
            label="Training NRMSE",
        )
        plt.plot(
            kld_indices,
            val_nrmse_list.to("cpu").detach().numpy(),
            marker="o",
            label="Validation NRMSE",
        )
        plt.xlabel("KLD Index")
        plt.ylabel("NRMSE")
        plt.title("NRMSE vs KLD Index")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(os.path.join(plot_folder, f"{file_name}_kld_nrmse.png"))
        # print(f"Best Model: {file_name} with KLD Index: {best_kld_index} ")

    ##### LOSS GRAPHS #####
    # fig1, axs1 = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column, optional figure size
    # # First subplot for Training Loss
    # axs1[0].plot((np.arange(len(losses)) +1) * epoch_plot_step, losses, label='Training Loss')
    # axs1[0].set_xlabel('Epochs')
    # axs1[0].set_ylabel('Loss')
    # axs1[0].set_title('Training Loss Curve')
    # axs1[0].legend()
    # # Second subplot for NRMSE
    # axs1[1].plot((np.arange(len(train_nrmse)) + 1) * epoch_plot_step, train_nrmse, label='NRMSE')
    # axs1[1].set_xlabel('Epochs')
    # axs1[1].set_ylabel('NRMSE')
    # axs1[1].set_title('NRMSE Curve')
    # axs1[1].legend()
    # plt.tight_layout()
    # fig1.savefig(os.path.join(plot_folder, f'{file_name}_lossgraph.png'))

    ####DEBUGGING PLOTS########
    # recon_output_ec, latent_space_mu, latent_space_logvar = model(Phi_rt_init, t1t2fatheta)
    # recon_output_ec = recon_output_ec.to('cpu').detach().numpy() # w/ eddy current
    # recon_output = model.decode(latent_space_mu, t1t2fatheta0) #NO REPARAMETERIZATION
    # recon_output = recon_output.to('cpu').detach().numpy()
    # latent_space_mu = latent_space_mu.to('cpu').detach().numpy()
    # np.save(os.path.join(assets_folder, f'{file_name}_latent_space.npy'), latent_space_mu)
    # np.save(os.path.join(assets_folder, f'{file_name}_recon_output.npy'), recon_output)
    # np.save(os.path.join(assets_folder, f'{file_name}_recon_output_ec.npy'), recon_output_ec)

    # output plots
    # fig2, axs2 = plt.subplots(2, 4, figsize=(20, 20))
    # axs2 = axs2.flatten()  # Flatten the 2x2 array of axes for easy indexing
    # for i in range(4):
    #     axs2[i].plot(Phi_rt_init[:3600, i].cpu(), label = f'Phi_rt_init Col {i+1}')
    #     axs2[i].plot(recon_output_ec[:3600, i], label = f'Recon_output w/eddy current Col {i+1}')
    #     axs2[i].legend(loc="upper right")
    # for i in range(4):
    #     axs2[i+4].plot(Phi_rt_init[:3600, i].cpu(), label=f'Phi_rt_init Col {i+1}')
    #     axs2[i+4].plot(recon_output[:3600, i], label=f'Recon_output Col {i+1}')
    #     axs2[i+4].legend(loc="upper right")
    # plt.tight_layout()
    # fig2.savefig(os.path.join(plot_folder, f'{file_name}_recon_output.png'))

    # latent vectors
    # fig3, axs3 = plt.subplots(1, 3, figsize=(30, 10))
    # # fig3, axs3 = plt.subplots(3, 1, figsize=(10, 10))
    # axs3 = axs3.flatten()  # Flatten the 2x2 array of axes for easy indexing
    # for i in range(3):
    #     axs3[i].plot(latent_space_mu[:4000, i], label=f'Latent vector {i+1}')
    #     axs3[i].legend(loc="upper right")
    # # axs3[3].plot(Segidx[:3600], label='Segidx')
    # # axs3[3].legend(loc="upper right")
    # # axs3[4].plot(wallClock[:3600], label='T2prep index')
    # # axs3[4].legend(loc="upper right")
    # # axs3[5].plot(flipAngles[:3600], label='Flip Angles')
    # # axs3[5].legend(loc="upper right")
    # plt.tight_layout()
    # fig3.savefig(os.path.join(plot_folder, f'{file_name}_latent_vector.png'))

    # fft of latent vectors
    # fig4, axs4 = plt.subplots(3, 1, figsize=(10, 10))
    # u2_fft = np.fft.fft(latent_space_mu[:,1])
    # u2_freq = np.fft.fftfreq(len(latent_space_mu[:,1]), d=0.0534)
    # u3_fft = np.fft.fft(latent_space_mu[:,2])
    # u3_freq = np.fft.fftfreq(len(latent_space_mu[:,2]), d=0.0534)
    # Phi_rt_init = Phi_rt_init.to('cpu').detach().numpy()
    # col8_fft = np.fft.fft(Phi_rt_init[:,7])
    # col8_freq = np.fft.fftfreq(len(Phi_rt_init[:,7]), d=0.0534)
    # axs4[0].plot(u2_freq, np.abs(u2_fft), label='fft of cardiac 1')
    # axs4[0].legend(loc="upper right")
    # axs4[1].plot(u3_freq, np.abs(u3_fft), label='fft of cardiac 2')
    # axs4[1].legend(loc="upper right")
    # axs4[2].plot(col8_freq, np.abs(col8_fft), label='fft of Phi_rt_init Col 8')
    # axs4[2].legend(loc="upper right")
    # plt.tight_layout()
    # fig4.savefig(os.path.join(plot_folder, f'{file_name}_fft_latent_vector.png'))


########### training ###########
# latent_dims = 2
# class_size = 2


# model = CVAE_BP(feature_size, latent_dims, class_size).to(device)

# optimizer = optim.Adam(model.parameters(), lr=1e-2)

# max_epochs = 10000
# epoch_plot_step = 10
# losses= []
# NRMSE = []

# print("training on", device)
# total = (Phi_rt_init**2).sum()
# pbar = tqdm(total=max_epochs, desc='Initializing')
# for epoch in range(max_epochs):
#     recon_x, mu, logvar = model(Phi_rt_init, t1t2_normalized)
#     optimizer.zero_grad()
#     loss = loss_function(recon_x, Phi_rt_init, mu, logvar)
#     loss.backward()
#     optimizer.step()
#     if (epoch % epoch_plot_step) == (epoch_plot_step - 1) :
#         train_loss = loss.detach().cpu().numpy()/Phi_rt_init.shape[0]
#         losses.append(train_loss)
#         mu, logvar = model.encode(Phi_rt_init, t1t2_normalized)
#         recon_x = model.decode(mu, t1t2_normalized) #NO REPARAMETERIZATION
#         mse = ((recon_x - Phi_rt_init)**2).sum()
#         nrmse_ = torch.sqrt(mse / total)
#         NRMSE.append(nrmse_.detach().cpu().numpy())
#         pbar.set_description(f'train_loss: {train_loss:.3e}, NRMSE: {nrmse_:.4f}')
#         pbar.update(epoch_plot_step)
# pbar.close()

# # Plot loss functions
# fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column, optional figure size
# # First subplot for Training Loss
# axs[0].plot((np.arange(len(losses)) +1) * epoch_plot_step, losses, label='Training Loss')
# axs[0].set_xlabel('Epochs')
# axs[0].set_ylabel('Loss')
# axs[0].set_title('Training Loss Curve')
# axs[0].legend()
# # Second subplot for NRMSE
# axs[1].plot((np.arange(len(NRMSE)) + 1) * epoch_plot_step, NRMSE, label='NRMSE')
# axs[1].set_xlabel('Epochs')
# axs[1].set_ylabel('NRMSE')
# axs[1].set_title('NRMSE Curve')
# axs[1].legend()
# plt.tight_layout()
# plt.savefig(os.path.join('./lossGraphs', 'losses_2inf.png'))

# print("Training complete")

# latent_space_mu, latent_space_logvar = model.encode(Phi_rt_init, t1t2_normalized)
# # without theta
# recon_output = model.decode(mu, t1t2_normalized) #NO REPARAMETERIZATION
# recon_output = recon_output.to('cpu').detach().numpy()
# np.save('./assets/recon_output_phantom.npy', recon_output)
# latent_space = model.reparameterize(latent_space_mu,latent_space_logvar)
# latent_space = latent_space.to('cpu').detach().numpy()
# np.save('./assets/latent_space_phantom.npy', latent_space)
# u1_fft = torch.fft.fft(latent_space_mu[:,0]).to('cpu').detach().numpy()
# u1_freq = torch.fft.fftfreq(len(latent_space_mu[:,0]), d=0.00712).to('cpu').detach().numpy()
# u2_fft = torch.fft.fft(latent_space_mu[:,1]).to('cpu').detach().numpy()
# u2_freq = torch.fft.fftfreq(len(latent_space_mu[:,1]), d=0.00712).to('cpu').detach().numpy()

# # Some plots. These should be in a different file at some point
# fig, axs = plt.subplots(5, 2, figsize=(10, 10))
# axs = axs.flatten()  # Flatten the 2x2 array of axes for easy indexing
# for i in range(4):
#     axs[i].plot(Phi_rt_init[:1800, i].cpu(), label='Phi_rt_init Col ' + str(i + 1))
#     axs[i].plot(recon_output[:1800, i], label='Recon_output Col ' + str(i + 1))
#     axs[i].legend(loc="upper right")
# for i in range(2):
#     axs[4+i].plot(latent_space_mu.detach().cpu().numpy()[:1800,i], label='Latent Col ' + str(i + 1))
#     axs[4+i].legend(loc="upper right")
# for i in range(2):
#     axs[6+i].plot(t1t2_normalized[:1800, i].cpu(), label='Known Latent Col ' + str(i + 1))
#     axs[6+i].legend(loc="upper right")
# axs[8].plot(u1_freq, np.abs(u1_fft), label='fft of Latent Col 1')
# axs[8].set_xlim(-4, 4)
# axs[8].legend(loc="upper right")
# axs[9].plot(u2_freq, np.abs(u2_fft), label='fft of Latent Col 2')
# axs[9].set_xlim(-4, 4)
# axs[9].legend(loc="upper right")
# # axs[9].plot(t1t2theta[:360,2].cpu(), label='theta')
# plt.tight_layout()
# plt.savefig('./plots/5x2_subplots_phantom.png')


### add thetas = 0 as the third parameter
# class_size = 4
# model = CVAE_BP_HT(feature_size, latent_dims, class_size).to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-2)

# max_epochs = 10000
# epoch_plot_step = 10
# losses= []
# NRMSE = []

# print("training on", device)
# total = (Phi_rt_init**2).sum()
# pbar = tqdm(total=max_epochs, desc='Initializing')
# for epoch in range(max_epochs):
#     recon_x, mu, logvar = model(Phi_rt_init, t1t2fatheta)
#     optimizer.zero_grad()
#     loss = loss_function(recon_x, Phi_rt_init, mu, logvar)
#     loss.backward()
#     optimizer.step()
#     if (epoch % epoch_plot_step) == (epoch_plot_step - 1) :
#         train_loss = loss.detach().cpu().numpy()/Phi_rt_init.shape[0]
#         losses.append(train_loss)
#         mu, logvar = model.encode(Phi_rt_init, t1t2fatheta)
#         recon_x = model.decode(mu, t1t2fatheta) #NO REPARAMETERIZATION
#         mse = ((recon_x - Phi_rt_init)**2).sum()
#         nrmse_ = torch.sqrt(mse / total)
#         NRMSE.append(nrmse_.detach().cpu().numpy())
#         pbar.set_description(f'train_loss: {train_loss:.3e}, NRMSE: {nrmse_:.4f}')
#         pbar.update(epoch_plot_step)
# pbar.close()

# torch.save(model.state_dict(), os.path.join('./SMS', 'sms_0528_run1.pth'))

# # Plot loss functions
# fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column, optional figure size
# # First subplot for Training Loss
# axs[0].plot((np.arange(len(losses)) +1) * epoch_plot_step, losses, label='Training Loss')
# axs[0].set_xlabel('Epochs')
# axs[0].set_ylabel('Loss')
# axs[0].set_title('Training Loss Curve')
# axs[0].legend()
# # Second subplot for NRMSE
# axs[1].plot((np.arange(len(NRMSE)) + 1) * epoch_plot_step, NRMSE, label='NRMSE')
# axs[1].set_xlabel('Epochs')
# axs[1].set_ylabel('NRMSE')
# axs[1].set_title('NRMSE Curve')
# axs[1].legend()
# plt.tight_layout()
# plt.savefig(os.path.join('./lossGraphs', 'sms.png'))

# print("Training complete")

# model.load_state_dict(torch.load('CVAEBP_HT.pth'))

# ###### pause contrast ######
# Segidx_max = Segidx_mu[119,:] #actually frame 120, not max value
# Segidx_fixed = Segidx_mu
# Segidx_fixed = Segidx_fixed.fill_(Segidx_max.item())

# wallClock_fixed = wallClock_mu
# wallClock_min = torch.min(wallClock_mu)
# wallClock_fixed = wallClock_fixed.fill_(wallClock_min.item())

# # NonlinearT1_max = torch.max(NonlinearT1_mu)
# # NonlinearT1_fixed = NonlinearT1_mu
# # NonlinearT1_fixed = NonlinearT1_fixed.fill_(NonlinearT1_max.item())

# # NonlinearT2_max = torch.max(NonlinearT2_mu)
# # NonlinearT2_fixed = NonlinearT2_mu
# # NonlinearT2_fixed = NonlinearT2_fixed.fill_(NonlinearT2_max.item())

# T1T2fixedtheta0 = torch.cat([Segidx_fixed, wallClock_fixed, theta0], dim=1).to(torch.float32)


# model.load_state_dict(torch.load('SMS/sms_0528_run1.pth'))

# latent_space_mu, latent_space_logvar = model.encode(Phi_rt_init, t1t2fatheta)
# recon_output = model.decode(latent_space_mu, t1t2fatheta0) #NO REPARAMETERIZATION
# recon_output = recon_output.to('cpu').detach().numpy()
# np.save('./SMS/recon_output_sms_run2.npy', recon_output)
# latent_space = model.reparameterize(latent_space_mu,latent_space_logvar)
# latent_space = latent_space.to('cpu').detach().numpy()
# u1_fft = torch.fft.fft(latent_space_mu[:,0]).to('cpu').detach().numpy()
# u1_freq = torch.fft.fftfreq(len(latent_space_mu[:,0]), d=0.00712).to('cpu').detach().numpy()
# u2_fft = torch.fft.fft(latent_space_mu[:,1]).to('cpu').detach().numpy()
# u2_freq = torch.fft.fftfreq(len(latent_space_mu[:,1]), d=0.00712).to('cpu').detach().numpy()
# u3_fft = torch.fft.fft(latent_space_mu[:,2]).to('cpu').detach().numpy()
# u3_freq = torch.fft.fftfreq(len(latent_space_mu[:,2]), d=0.00712).to('cpu').detach().numpy()
# # np.save('./assets/latent_space_CVAEBPHT_w_eddy.npy', latent_space)

# fig, axs = plt.subplots(7, 2, figsize=(10, 10))
# axs = axs.flatten()  # Flatten the 2x2 array of axes for easy indexing
# for i in range(4):
#     axs[i].plot(Phi_rt_init[:1800, i].cpu(), label='Phi_rt_init Col ' + str(i + 1))
#     axs[i].plot(recon_output[:1800, i], label='Recon_output Col ' + str(i + 1))
#     axs[i].legend(loc="upper right")
# for i in range(3):
#     axs[4+i].plot(latent_space_mu.detach().cpu().numpy()[:1800,i], label='Latent Col ' + str(i + 1))
#     axs[4+i].legend(loc="upper right")
# axs[7].plot(u1_freq, np.abs(u1_fft), label='fft of Latent Col 1')
# axs[7].set_xlim(-4, 4)
# axs[7].legend(loc="upper right")
# axs[8].plot(u2_freq, np.abs(u2_fft), label='fft of Latent Col 2')
# axs[8].set_xlim(-4, 4)
# axs[8].legend(loc="upper right")
# axs[9].plot(u3_freq, np.abs(u3_fft), label='fft of Latent Col 3')
# axs[9].set_xlim(-4, 4)
# axs[9].legend(loc="upper right")
# for i in range(4):
#     axs[10+i].plot(t1t2fatheta[:1800, i].cpu(), label='Known Latent Col ' + str(i + 1))
#     axs[10+i].legend(loc="upper right")
# plt.tight_layout()
# plt.savefig('./SMS/7x2_subplots_0528_run1.png')

# latent_space_mu, latent_space_logvar = model.encode(Phi_rt_init, t1t2theta)
# recon_output = model.decode(latent_space_mu, T1T2fixedtheta0) #NO REPARAMETERIZATION
# recon_output = recon_output.to('cpu').detach().numpy()
# np.save('./assets/recon_output_CVAEBP_HT_wo_eddy_pausecontrast.npy', recon_output)
# latent_space = model.reparameterize(latent_space_mu,latent_space_logvar)
# latent_space = latent_space.to('cpu').detach().numpy()
# u1_fft = torch.fft.fft(latent_space_mu[:,0]).to('cpu').detach().numpy()
# u1_freq = torch.fft.fftfreq(len(latent_space_mu[:,0]), d=0.00712).to('cpu').detach().numpy()
# u2_fft = torch.fft.fft(latent_space_mu[:,1]).to('cpu').detach().numpy()
# u2_freq = torch.fft.fftfreq(len(latent_space_mu[:,1]), d=0.00712).to('cpu').detach().numpy()
# u3_fft = torch.fft.fft(latent_space_mu[:,2]).to('cpu').detach().numpy()
# u3_freq = torch.fft.fftfreq(len(latent_space_mu[:,2]), d=0.00712).to('cpu').detach().numpy()
# np.save('./assets/latent_space_CVAEBP_HT_wo_eddy_pausecontrast.npy', latent_space)

# fig, axs = plt.subplots(5, 2, figsize=(10, 10))
# axs = axs.flatten()  # Flatten the 2x2 array of axes for easy indexing
# for i in range(4):
#     axs[i].plot(Phi_rt_init[:1800, i].cpu(), label='Phi_rt_init Col ' + str(i + 1))
#     axs[i].plot(recon_output[:1800, i], label='Recon_output Col ' + str(i + 1))
#     axs[i].legend(loc="upper right")
# for i in range(3):
#     axs[4+i].plot(latent_space_mu.detach().cpu().numpy()[:1800,i], label='Latent Col ' + str(i + 1))
#     axs[4+i].legend(loc="upper right")
# axs[7].plot(u1_freq, np.abs(u1_fft), label='fft of Latent Col 1')
# axs[7].set_xlim(-4, 4)
# axs[7].legend(loc="upper right")
# axs[8].plot(u2_freq, np.abs(u2_fft), label='fft of Latent Col 2')
# axs[8].set_xlim(-4, 4)
# axs[8].legend(loc="upper right")
# axs[9].plot(u3_freq, np.abs(u3_fft), label='fft of Latent Col 3')
# axs[9].set_xlim(-4, 4)
# axs[9].legend(loc="upper right")
# plt.tight_layout()
# plt.savefig('./plots/5x2_subplots_CVAEBPHT_pausecontrast.png')

# latent_space_mu, latent_space_logvar = model.encode(Phi_rt_init, t1t2theta)
# ###pause motion###
# latent_space_mu_pause = torch.ones(latent_space_mu.shape).to(device)
# latent_space_mu_pause[:,1] = 4 * latent_space_mu_pause[:,1].to(device)
# latent_space_mu_pause[:,2] = 5 * latent_space_mu_pause[:,2].to(device)
# recon_output = model.decode(latent_space_mu_pause, t1t2theta0) #NO REPARAMETERIZATION
# recon_output = recon_output.to('cpu').detach().numpy()
# np.save('./assets/recon_output_CVAEBP_HT_wo_eddy_pausemotion.npy', recon_output)
# latent_space = model.reparameterize(latent_space_mu,latent_space_logvar)
# latent_space = latent_space.to('cpu').detach().numpy()
# u1_fft = torch.fft.fft(latent_space_mu[:,0]).to('cpu').detach().numpy()
# u1_freq = torch.fft.fftfreq(len(latent_space_mu[:,0]), d=0.00712).to('cpu').detach().numpy()
# u2_fft = torch.fft.fft(latent_space_mu[:,1]).to('cpu').detach().numpy()
# u2_freq = torch.fft.fftfreq(len(latent_space_mu[:,1]), d=0.00712).to('cpu').detach().numpy()
# u3_fft = torch.fft.fft(latent_space_mu[:,2]).to('cpu').detach().numpy()
# u3_freq = torch.fft.fftfreq(len(latent_space_mu[:,2]), d=0.00712).to('cpu').detach().numpy()
# np.save('./assets/latent_space_CVAEBP_HT_wo_eddy_pausemotion.npy', latent_space)


# ###resp end of expiration; cardiac realtime###
# latent_space_mu, latent_space_logvar = model.encode(Phi_rt_init, t1t2theta)
# latent_space_mu_ee = latent_space_mu
# latent_space_mu_ee[:,0] = -1.2 * torch.ones(latent_space_mu[:,0].shape).to(device)
# recon_output = model.decode(latent_space_mu_ee, T1T2fixedtheta0) #NO REPARAMETERIZATION
# recon_output = recon_output.to('cpu').detach().numpy()
# np.save('./assets/recon_output_CVAEBP_HT_ee_card_rt.npy', recon_output)
# latent_space = model.reparameterize(latent_space_mu,latent_space_logvar)
# latent_space = latent_space.to('cpu').detach().numpy()
# u1_fft = torch.fft.fft(latent_space_mu[:,0]).to('cpu').detach().numpy()
# u1_freq = torch.fft.fftfreq(len(latent_space_mu[:,0]), d=0.00712).to('cpu').detach().numpy()
# u2_fft = torch.fft.fft(latent_space_mu[:,1]).to('cpu').detach().numpy()
# u2_freq = torch.fft.fftfreq(len(latent_space_mu[:,1]), d=0.00712).to('cpu').detach().numpy()
# u3_fft = torch.fft.fft(latent_space_mu[:,2]).to('cpu').detach().numpy()
# u3_freq = torch.fft.fftfreq(len(latent_space_mu[:,2]), d=0.00712).to('cpu').detach().numpy()
# np.save('./assets/latent_space_CVAEBP_HT_ee_card_rt.npy', latent_space)

# ###resp: end of expiration; cardiac: sinusoidal signal with 50 phases###
# latent_space_mu, latent_space_logvar = model.encode(Phi_rt_init, t1t2theta)
# latent_space_mu_ee = latent_space_mu
# latent_space_mu_ee[:,0] = -1.2 * torch.ones(latent_space_mu[:,0].shape).to(device)
# ## average value of latent_space_mu[:,1] and latent_space_mu[:,2]
# aver_cardiac = torch.sqrt(latent_space_mu[:,1]**2 + latent_space_mu[:,2]**2).to(device)
# t_cardiac1 = torch.linspace(0, latent_space_mu[:,0].shape[0] - 1, latent_space_mu[:,0].shape[0]).to(device)
# t_cardiac2 = torch.linspace(0, latent_space_mu[:,0].shape[0] - 1, latent_space_mu[:,0].shape[0]).to(device)
# phase_num = 50
# frequency = 1 / phase_num
# cardiac1 = torch.cos(2 * torch.pi * frequency * t_cardiac1).to(device)
# cardiac2 = torch.sin(2 * torch.pi * frequency * t_cardiac2).to(device)
# latent_space_mu_ee[:,1] = aver_cardiac * cardiac1.to(device)
# latent_space_mu_ee[:,2] = aver_cardiac * cardiac2.to(device)
# recon_output = model.decode(latent_space_mu_ee, T1T2fixedtheta0) #NO REPARAMETERIZATION
# recon_output = recon_output.to('cpu').detach().numpy()
# np.save('./assets/recon_output_CVAEBP_HT_ee_card_sin.npy', recon_output)
# latent_space = model.reparameterize(latent_space_mu,latent_space_logvar)
# latent_space = latent_space.to('cpu').detach().numpy()
# u1_fft = torch.fft.fft(latent_space_mu[:,0]).to('cpu').detach().numpy()
# u1_freq = torch.fft.fftfreq(len(latent_space_mu[:,0]), d=0.00712).to('cpu').detach().numpy()
# u2_fft = torch.fft.fft(latent_space_mu[:,1]).to('cpu').detach().numpy()
# u2_freq = torch.fft.fftfreq(len(latent_space_mu[:,1]), d=0.00712).to('cpu').detach().numpy()
# u3_fft = torch.fft.fft(latent_space_mu[:,2]).to('cpu').detach().numpy()
# u3_freq = torch.fft.fftfreq(len(latent_space_mu[:,2]), d=0.00712).to('cpu').detach().numpy()
# np.save('./assets/latent_space_CVAEBP_HT_ee_card_sin.npy', latent_space)

# ###resp: end of expiration; cardiac: end of diastole###
# latent_space_mu, latent_space_logvar = model.encode(Phi_rt_init, t1t2theta)
# latent_space_mu_ed = latent_space_mu
# latent_space_mu_ed[:,0] = -1.2 * torch.ones(latent_space_mu[:,0].shape).to(device)
# ### fixing end of diastole at timepoint 144
# latent_space_mu_ed[:,1] = latent_space_mu[144,1] * torch.ones(latent_space_mu[:,1].shape).to(device)
# latent_space_mu_ed[:,2] = latent_space_mu[144,2] * torch.ones(latent_space_mu[:,2].shape).to(device)
# recon_output = model.decode(latent_space_mu_ed, t1t2theta0) #NO REPARAMETERIZATION
# recon_output = recon_output.to('cpu').detach().numpy()
# np.save('./assets/recon_output_CVAEBP_HT_ee_ed.npy', recon_output)
# latent_space = model.reparameterize(latent_space_mu,latent_space_logvar)
# latent_space = latent_space.to('cpu').detach().numpy()
# u1_fft = torch.fft.fft(latent_space_mu[:,0]).to('cpu').detach().numpy()
# u1_freq = torch.fft.fftfreq(len(latent_space_mu[:,0]), d=0.00712).to('cpu').detach().numpy()
# u2_fft = torch.fft.fft(latent_space_mu[:,1]).to('cpu').detach().numpy()
# u2_freq = torch.fft.fftfreq(len(latent_space_mu[:,1]), d=0.00712).to('cpu').detach().numpy()
# u3_fft = torch.fft.fft(latent_space_mu[:,2]).to('cpu').detach().numpy()
# u3_freq = torch.fft.fftfreq(len(latent_space_mu[:,2]), d=0.00712).to('cpu').detach().numpy()
# np.save('./assets/latent_space_CVAEBP_HT_ee_ed.npy', latent_space)

# ###resp: end of inhalation to end of exhale; cardiac: end of diastole###
# latent_space_mu, latent_space_logvar = model.encode(Phi_rt_init, t1t2theta)
# latent_space_mu_resp_phase = latent_space_mu
# resp = torch.linspace(1.2, -1.2, 20)
# # repeat resp to match the total timepoints
# latent_space_mu_resp_phase[:,0] = resp.repeat(latent_space_mu[:,0].shape[0] // 20 + 1)[:latent_space_mu[:,0].shape[0]].to(device)
# ### fixing end of diastole at timepoint 144
# latent_space_mu_resp_phase[:,1] = latent_space_mu[144,1] * torch.ones(latent_space_mu[:,1].shape).to(device)
# latent_space_mu_resp_phase[:,2] = latent_space_mu[144,2] * torch.ones(latent_space_mu[:,2].shape).to(device)
# recon_output = model.decode(latent_space_mu_resp_phase, T1T2fixedtheta0) #NO REPARAMETERIZATION
# recon_output = recon_output.to('cpu').detach().numpy()
# np.save('./assets/recon_output_CVAEBP_HT_resp_phase.npy', recon_output)
# latent_space = model.reparameterize(latent_space_mu,latent_space_logvar)
# latent_space = latent_space.to('cpu').detach().numpy()
# u1_fft = torch.fft.fft(latent_space_mu[:,0]).to('cpu').detach().numpy()
# u1_freq = torch.fft.fftfreq(len(latent_space_mu[:,0]), d=0.00712).to('cpu').detach().numpy()
# u2_fft = torch.fft.fft(latent_space_mu[:,1]).to('cpu').detach().numpy()
# u2_freq = torch.fft.fftfreq(len(latent_space_mu[:,1]), d=0.00712).to('cpu').detach().numpy()
# u3_fft = torch.fft.fft(latent_space_mu[:,2]).to('cpu').detach().numpy()
# u3_freq = torch.fft.fftfreq(len(latent_space_mu[:,2]), d=0.00712).to('cpu').detach().numpy()
# np.save('./assets/latent_space_CVAEBP_HT_resp_phase.npy', latent_space)


# fig, axs = plt.subplots(5, 2, figsize=(10, 10))
# axs = axs.flatten()  # Flatten the 2x2 array of axes for easy indexing
# for i in range(4):
#     axs[i].plot(Phi_rt_init[:1800, i].cpu(), label='Phi_rt_init Col ' + str(i + 1))
#     axs[i].plot(recon_output[:1800, i], label='Recon_output Col ' + str(i + 1))
#     axs[i].legend(loc="upper right")
# for i in range(3):
#     axs[4+i].plot(latent_space_mu.detach().cpu().numpy()[:1800,i], label='Latent Col ' + str(i + 1))
#     axs[4+i].legend(loc="upper right")
# axs[7].plot(u1_freq, np.abs(u1_fft), label='fft of Latent Col 1')
# axs[7].set_xlim(-4, 4)
# axs[7].legend(loc="upper right")
# axs[8].plot(u2_freq, np.abs(u2_fft), label='fft of Latent Col 2')
# axs[8].set_xlim(-4, 4)
# axs[8].legend(loc="upper right")
# axs[9].plot(u3_freq, np.abs(u3_fft), label='fft of Latent Col 3')
# axs[9].set_xlim(-4, 4)
# axs[9].legend(loc="upper right")
# plt.tight_layout()
# plt.savefig('./plots/5x2_subplots_CVAEBPHT_pausemotion.png')

# fig, axs = plt.subplots(5, 2, figsize=(10, 10))
# axs = axs.flatten()  # Flatten the 2x2 array of axes for easy indexing
# for i in range(4):
#     axs[i].plot(Phi_rt_init[:1800, i].cpu(), label='Phi_rt_init Col ' + str(i + 1))
#     axs[i].plot(recon_output[:1800, i], label='Recon_output Col ' + str(i + 1))
#     axs[i].legend(loc="upper right")
# for i in range(2):
#     axs[4+i].plot(latent_space_mu.detach().cpu().numpy()[:1800,i], label='Latent Col ' + str(i + 1))
#     axs[4+i].legend(loc="upper right")
# for i in range(2):
#     axs[6+i].plot(t1t2theta0[:1800, i].cpu(), label='Known Latent Col ' + str(i + 1))
#     axs[6+i].legend(loc="upper right")
# axs[8].plot(u1_freq, np.abs(u1_fft), label='fft of Latent Col 1')
# axs[8].set_xlim(-4, 4)
# axs[8].legend(loc="upper right")
# axs[9].plot(u2_freq, np.abs(u2_fft), label='fft of Latent Col 2')
# axs[9].set_xlim(-4, 4)
# axs[9].legend(loc="upper right")
# plt.tight_layout()
# plt.savefig('./plots/5x2_subplots_CVAEBP_25_wo_eddy_pausemotion.png')


# torchaudio as filtering
# class_size = 3
# model = CVAE(feature_size, latent_dims, class_size).to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-2)
# sampling_rate = 140.45

# max_epochs = 10000
# epoch_plot_step = 10
# losses= []
# NRMSE = []

# print("training on", device)
# total = (Phi_rt_init**2).sum()
# pbar = tqdm(total=max_epochs, desc='Initializing')
# for epoch in range(max_epochs):
#     recon_x, mu, logvar = model(Phi_rt_init, t1t2theta)
#     u1 = mu[:,0]
#     u2 = mu[:,1]
#     u1 = torchaudio.functional.lowpass_biquad(u1, 140, 0.67)
#     u2 = torchaudio.functional.bandpass_biquad(u2, 140, 1.42, 0.608)
#     mu_filtered = torch.stack((u1,u2),dim=1).real
#     optimizer.zero_grad()
#     loss = loss_function(recon_x, Phi_rt_init, mu, mu_filtered, logvar)
#     loss.backward()
#     optimizer.step()
#     if (epoch % epoch_plot_step) == (epoch_plot_step - 1) :
#         train_loss = loss.detach().cpu().numpy()/Phi_rt_init.shape[0]
#         losses.append(train_loss)
#         recon_x = model.decode(mu, t1t2theta) #NO REPARAMETERIZATION
#         mse = ((recon_x - Phi_rt_init)**2).sum()
#         nrmse_ = torch.sqrt(mse / total)
#         NRMSE.append(nrmse_.detach().cpu().numpy())
#         pbar.set_description(f'train_loss: {train_loss:.3e}, NRMSE: {nrmse_:.4f}')
#         pbar.update(epoch_plot_step)
# pbar.close()

# # Plot loss functions
# fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column, optional figure size
# # First subplot for Training Loss
# axs[0].plot((np.arange(len(losses)) +1) * epoch_plot_step, losses, label='Training Loss')
# axs[0].set_xlabel('Epochs')
# axs[0].set_ylabel('Loss')
# axs[0].set_title('Training Loss Curve')
# axs[0].legend()
# # Second subplot for NRMSE
# axs[1].plot((np.arange(len(NRMSE)) + 1) * epoch_plot_step, NRMSE, label='NRMSE')
# axs[1].set_xlabel('Epochs')
# axs[1].set_ylabel('NRMSE')
# axs[1].set_title('NRMSE Curve')
# axs[1].legend()
# plt.tight_layout()
# plt.savefig(os.path.join('./lossGraphs', 'losses_BPF_audio_e-6.png'))

# print("Training complete")

# latent_space_mu, latent_space_logvar = model.encode(Phi_rt_init, t1t2theta)
# recon_output = model.decode(latent_space_mu, t1t2theta0) #NO REPARAMETERIZATION
# recon_output = recon_output.to('cpu').detach().numpy()
# np.save('./assets/recon_output_BPF_audio_e-6.npy', recon_output)
# latent_space = model.reparameterize(latent_space_mu,latent_space_logvar)
# latent_space = latent_space.to('cpu').detach().numpy()
# np.save('./assets/latent_space_BPF_audio_e-6.npy', latent_space)

# fig, axs = plt.subplots(5, 2, figsize=(10, 10))
# axs = axs.flatten()  # Flatten the 2x2 array of axes for easy indexing
# for i in range(4):
#     axs[i].plot(Phi_rt_init[:1800, i].cpu(), label='Phi_rt_init Col ' + str(i + 1))
#     axs[i].plot(recon_output[:1800, i], label='Recon_output Col ' + str(i + 1))
#     axs[i].legend(loc="upper right")
# for i in range(2):
#     axs[4+i].plot(latent_space_mu.detach().cpu().numpy()[:1800,i], label='Latent Col ' + str(i + 1))
#     axs[4+i].legend(loc="upper right")
# for i in range(3):
#     axs[6+i].plot(t1t2theta0[:1800, i].cpu(), label='Known Latent Col ' + str(i + 1))
#     axs[6+i].legend(loc="upper right")
# axs[9].plot(t1t2theta[:360,2].cpu(), label='theta')
# plt.tight_layout()
# plt.savefig('./plots/5x2_subplots_BPF_audio_e-6.png')


# #### add thetas as the third parameter
# class_size = 3
# model1 = CVAE(feature_size, latent_dims, class_size).to(device)
# optimizer = optim.Adam(model1.parameters(), lr=1e-2)

# max_epochs = 10000
# epoch_plot_step = 10
# losses= []
# NRMSE = []

# print("training on", device)
# total = (Phi_rt_init**2).sum()
# pbar = tqdm(total=max_epochs, desc='Initializing')
# for epoch in range(max_epochs):
#     recon_x, mu, logvar = model1(Phi_rt_init, t1t2theta)
#     optimizer.zero_grad()
#     loss = loss_function(recon_x, Phi_rt_init, mu, logvar)
#     loss.backward()
#     optimizer.step()
#     if (epoch % epoch_plot_step) == (epoch_plot_step - 1) :
#         train_loss = loss.detach().cpu().numpy()/Phi_rt_init.shape[0]
#         losses.append(train_loss)
#         mu, logvar = model1.encode(Phi_rt_init, t1t2theta)
#         recon_x = model1.decode(mu, t1t2theta) #NO REPARAMETERIZATION
#         mse = ((recon_x - Phi_rt_init)**2).sum()
#         nrmse_ = torch.sqrt(mse / total)
#         NRMSE.append(nrmse_.detach().cpu().numpy())
#         pbar.set_description(f'train_loss: {train_loss:.3e}, NRMSE: {nrmse_:.4f}')
#         pbar.update(epoch_plot_step)
# pbar.close()

# # Plot loss functions
# fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column, optional figure size
# # First subplot for Training Loss
# axs[0].plot((np.arange(len(losses)) +1) * epoch_plot_step, losses, label='Training Loss')
# axs[0].set_xlabel('Epochs')
# axs[0].set_ylabel('Loss')
# axs[0].set_title('Training Loss Curve')
# axs[0].legend()
# # Second subplot for NRMSE
# axs[1].plot((np.arange(len(NRMSE)) + 1) * epoch_plot_step, NRMSE, label='NRMSE')
# axs[1].set_xlabel('Epochs')
# axs[1].set_ylabel('NRMSE')
# axs[1].set_title('NRMSE Curve')
# axs[1].legend()
# plt.tight_layout()
# plt.savefig(os.path.join('./lossGraphs', 'losses_3inf.png'))

# print("Training complete")

# latent_space_mu, latent_space_logvar = model1.encode(Phi_rt_init, t1t2theta)
# # with theta
# recon_output = model1.decode(mu, t1t2theta0) #NO REPARAMETERIZATION
# recon_output = recon_output.to('cpu').detach().numpy()
# np.save('./assets/recon_output_stemberg_3inf_theta0.npy', recon_output)
# latent_space = model1.reparameterize(latent_space_mu,latent_space_logvar)
# latent_space = latent_space.to('cpu').detach().numpy()
# np.save('./assets/latent_space_stemberg_3inf_theta0.npy', latent_space)

# fig, axs = plt.subplots(5, 2, figsize=(10, 10))
# axs = axs.flatten()  # Flatten the 2x2 array of axes for easy indexing
# for i in range(4):
#     axs[i].plot(Phi_rt_init[:1800, i].cpu(), label='Phi_rt_init Col ' + str(i + 1))
#     axs[i].plot(recon_output[:1800, i], label='Recon_output Col ' + str(i + 1))
#     axs[i].legend(loc="upper right")
# for i in range(2):
#     axs[4+i].plot(latent_space_mu.detach().cpu().numpy()[:1800,i], label='Latent Col ' + str(i + 1))
#     axs[4+i].legend(loc="upper right")
# for i in range(3):
#     axs[6+i].plot(t1t2theta[:1800, i].cpu(), label='Known Latent Col ' + str(i + 1))
#     axs[6+i].legend(loc="upper right")
# axs[9].plot(t1t2theta[:360,2].cpu(), label='theta')
# plt.tight_layout()
# plt.savefig('./plots/5x2_subplots_stemberg_3inf_theta0.png')

# latent_space = np.load('./assets/latent_space_CVAEBPHT_wo_eddy.npy')
# print(latent_space.shape)
# u2 = latent_space[:360,1]
# u3 = latent_space[:360,2]
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(u2, u3)
# ax.spines['left'].set_position('zero')
# ax.spines['bottom'].set_position('zero')
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
# ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
# ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
# ax.set_title('Plot of cardiac circle')
# ax.grid(True)
# plt.savefig('./plots/cardiac_circle.png')

# amplitude_average = np.mean(np.sqrt(u2**2 + u3**2))
# u2 = amplitude_average * np.cos(np.linspace(-np.pi, np.pi, 1800))
# u3 = amplitude_average * np.sin(np.linspace(-np.pi, np.pi, 1800))

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(u2, u3)
# ax.spines['left'].set_position('zero')
# ax.spines['bottom'].set_position('zero')
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
# ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
# ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
# ax.set_title('Plot of cardiac circle')
# ax.grid(True)
# plt.savefig('./plots/cardiac_circle_aver.png')


# # plot the distribution of latent space
# latent_space = np.load('./assets/latent_space_CVAEBPHT_wo_eddy.npy')
# resp = latent_space[:,0]
# fig = plt.hist(resp, bins=100, edgecolor='black')
# plt.title('Respiratory Distribution')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.savefig('./plots/resp_distribution.png')


# ### DAE FOR DISENTANGLEMENT ###
# latent_dims = 2
# class_size = 3
# feature_size = Phi_rt_init.size()[1]
# model = DAE(feature_size, latent_dims, class_size).to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-2)

# max_epochs = 10000
# epoch_plot_step = 10
# losses= []
# NRMSE = []

# print("training on", device)
# total = (Phi_rt_init**2).sum()
# pbar = tqdm(total=max_epochs, desc='Initializing')
# for epoch in range(max_epochs):
#     recon_x, mu, logvar = model(Phi_rt_init, t1t2theta)
#     optimizer.zero_grad()
#     loss = loss_function(recon_x, Phi_rt_init, mu, logvar)
#     loss.backward()
#     optimizer.step()
#     if (epoch % epoch_plot_step) == (epoch_plot_step - 1) :
#         train_loss = loss.detach().cpu().numpy()/Phi_rt_init.shape[0]
#         losses.append(train_loss)
#         mu, logvar = model.encode(Phi_rt_init, t1t2theta)
#         recon_x = model.decode(mu, t1t2theta) #NO REPARAMETERIZATION
#         mse = ((recon_x - Phi_rt_init)**2).sum()
#         nrmse_ = torch.sqrt(mse / total)
#         NRMSE.append(nrmse_.detach().cpu().numpy())
#         pbar.set_description(f'train_loss: {train_loss:.3e}, NRMSE: {nrmse_:.4f}')
#         pbar.update(epoch_plot_step)
# pbar.close()

# torch.save(model.state_dict(), os.path.join('./results', 'dae_test.pth'))

# fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column, optional figure size
# # First subplot for Training Loss
# axs[0].plot((np.arange(len(losses)) +1) * epoch_plot_step, losses, label='Training Loss')
# axs[0].set_xlabel('Epochs')
# axs[0].set_ylabel('Loss')
# axs[0].set_title('Training Loss Curve')
# axs[0].legend()
# # Second subplot for NRMSE
# axs[1].plot((np.arange(len(NRMSE)) + 1) * epoch_plot_step, NRMSE, label='NRMSE')
# axs[1].set_xlabel('Epochs')
# axs[1].set_ylabel('NRMSE')
# axs[1].set_title('NRMSE Curve')
# axs[1].legend()
# plt.tight_layout()
# plt.savefig(os.path.join('./lossGraphs', 'dae.png'))

# print("Training complete")

# latent_space_mu, latent_space_logvar = model.encode(Phi_rt_init, t1t2theta)
# recon_output = model.decode(latent_space_mu, t1t2theta) #NO REPARAMETERIZATION
# recon_output = recon_output.to('cpu').detach().numpy()
# np.save('./assets/recon_output_dae.npy', recon_output)
# latent_space = model.reparameterize(latent_space_mu,latent_space_logvar)
# latent_space = latent_space.to('cpu').detach().numpy()
# np.save('./assets/latent_space_dae.npy', latent_space)
# u1_fft = torch.fft.fft(latent_space_mu[:,0]).to('cpu').detach().numpy()
# u1_freq = torch.fft.fftfreq(len(latent_space_mu[:,0]), d=0.00712).to('cpu').detach().numpy()
# u2_fft = torch.fft.fft(latent_space_mu[:,1]).to('cpu').detach().numpy()
# u2_freq = torch.fft.fftfreq(len(latent_space_mu[:,1]), d=0.00712).to('cpu').detach().numpy()

# fig, axs = plt.subplots(5, 2, figsize=(10, 10))
# axs = axs.flatten()  # Flatten the 2x2 array of axes for easy indexing
# for i in range(4):
#     axs[i].plot(Phi_rt_init[:1800, i].cpu(), label='Phi_rt_init Col ' + str(i + 1))
#     axs[i].plot(recon_output[:1800, i], label='Recon_output Col ' + str(i + 1))
#     axs[i].legend(loc="upper right")
# for i in range(2):
#     axs[4+i].plot(latent_space_mu.detach().cpu().numpy()[:1800,i], label='Latent Col ' + str(i + 1))
#     axs[4+i].legend(loc="upper right")
# for i in range(2):
#     axs[6+i].plot(t1t2_normalized[:1800, i].cpu(), label='Known Latent Col ' + str(i + 1))
#     axs[6+i].legend(loc="upper right")
# axs[8].plot(u1_freq, np.abs(u1_fft), label='fft of Latent Col 1')
# axs[8].set_xlim(-4, 4)
# axs[8].legend(loc="upper right")
# axs[9].plot(u2_freq, np.abs(u2_fft), label='fft of Latent Col 2')
# axs[9].set_xlim(-4, 4)
# axs[9].legend(loc="upper right")
# # axs[9].plot(t1t2theta[:360,2].cpu(), label='theta')
# plt.tight_layout()
# plt.savefig('./plots/5x2_subplots_dae.png')
