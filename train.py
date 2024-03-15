import os
import sys
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils
import torch.distributions
import torch.optim as optim
import time
from tqdm import tqdm
from network import VAE, CVAE
# from Vae1 import vae_test2, vae_test1, vae_test3, vae_test4, vae_test5, vae_test6, vae_test8
#from models import VariationalAutoencoder_hc, VariationalAutoencoder_hc_m, VariationalAutoencoder_linear4, VariationalAutoencoder_nonlinear4
import torchvision
import numpy as np
import scipy
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from sklearn.feature_selection import mutual_info_regression
from torch.utils.tensorboard import SummaryWriter
#Configuration
# device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# MI = MutualInformation(num_bins=256, sigma=0.4, normalize=True).to(device)

#data_init

path = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(path, 'binning')
data_file = os.path.join(data_folder, 'realtime_GT_imgs.mat')
mat = scipy.io.loadmat(data_file)
Phi_rt_init = mat['Phi_rt_init'] 
Phi_rt_init_real = Phi_rt_init.real
Phi_rt_init_imag = Phi_rt_init.imag
Phi_rt_init_con = np.concatenate((Phi_rt_init_real, Phi_rt_init_imag))
Phi_rt_init_con = Phi_rt_init_con.transpose()
Phi_rt_init = torch.from_numpy(Phi_rt_init_con).float().to(device)

#data_feature
data_file= os.path.join(data_folder, 'feature.mat')
mat = scipy.io.loadmat(data_file)
Phi_rt_feature = mat['Phi_rt_feature'] 
Phi_rt_feature_real = Phi_rt_feature.real
Phi_rt_feature_imag = Phi_rt_feature.imag
Phi_rt_feature_con = np.concatenate((Phi_rt_feature_real, Phi_rt_feature_imag))
Phi_rt_feature_con = Phi_rt_feature_con.transpose()
Phi_rt_feature = torch.from_numpy(Phi_rt_feature_con).float().to(device)

# print(data_con.shape)

seg_file = os.path.join(data_folder, 'TI.mat')
segidx = scipy.io.loadmat(seg_file)
segidx = segidx['Segidx']

binning_file = os.path.join(data_folder, 'binning_GT.mat')
binning_file = scipy.io.loadmat(binning_file)
hidx = binning_file['Hidx'] # Cardiac
ridx = binning_file['Ridx'] # Respiratory

# create simulated t2 prep signal
array_1 = np.ones(360)
array_2 = 2 * array_1
array_3 = 3 * array_1
array_4 = 4 * array_1
array_5 = 5 * array_1
array = np.concatenate((array_1, array_2, array_3),axis=0)
array = np.concatenate((array, array_4, array_5),axis=0)
array = np.tile(array, 7)

ridx_mu = (ridx - ridx.mean())/ridx.std()
hidx_mu = (hidx - hidx.mean())/hidx.std()
segidx_mu = (segidx - segidx.mean())/segidx.std()
ramp_latent_mu = (array-array.mean())/array.std()

ramp_latent_mu = torch.from_numpy(ramp_latent_mu).unsqueeze(1).to(device)
segidx_mu  = torch.from_numpy(segidx_mu).to(device)
t1t2_normalized = torch.cat([segidx_mu,ramp_latent_mu], dim=1).to(torch.float32)

class CustomDataset(Dataset):
    def __init__(self, data_feature, data_t1t2):
        self.data = data_feature
        self.t1t2 = data_t1t2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the data and t1t2 for the given index
        data_sample = self.data[index]
        t1t2_sample = self.t1t2[index]

        return data_sample, t1t2_sample

train_dataset = CustomDataset(Phi_rt_feature, t1t2_normalized)
train_dataset_init = CustomDataset(Phi_rt_init, t1t2_normalized)

#####
# def train(autoencoder, data, epochs=1):
#     opt = torch.optim.Adam(autoencoder.parameters())
#     # recon_output = torch.zeros(12600,96) #######
#     # train_losses = torch.zeros(epochs,1)
#     for epoch in range(epochs):
#         total_loss = 0.0
#         for idx, x in enumerate(data):
#             x = x.to(device) 
#             opt.zero_grad()
#             x_hat = autoencoder(x,idx)
#             # recon_output[idx] = x_hat
#             loss =  ((x - x_hat)**2).sum()  + autoencoder.encoder.kl
#             kl = autoencoder.encoder.kl
#             # MSELoss = ((x - x_hat)**2).sum()  
#             # total = (x*x).sum()
#             # total_loss = loss
#             loss.backward()
#             opt.step()
#         # train_losses[epoch] = total_loss
#     return autoencoder, loss, kl #total, recon_output, train_losses

# def train_test(autoencoder, data, epochs=200):
#     opt = torch.optim.Adam(autoencoder.parameters())
#     train_losses = torch.zeros(epochs,1)
#     for epoch in range(epochs):
#         autoencoder.train()
#         total_loss = 0.0
#         for idx, x in enumerate(data):
#             x = x.to(device) 
#             opt.zero_grad()
#             x_hat = autoencoder(x,idx)
#             MSELoss = ((x - x_hat)**2).sum() 
#             loss =  ((x - x_hat)**2).sum()  + autoencoder.encoder.kl 
#             total_loss = loss
#             total = (x**2).sum()
#             loss.backward()
#             opt.step()
#         train_losses[epoch] = total_loss
#     return autoencoder, MSELoss, train_losses

# def train_test_short(autoencoder, data, epochs=20):
#     opt = torch.optim.Adam(autoencoder.parameters())
#     train_losses = torch.zeros(epochs,1)
#     for epoch in range(epochs):
#         autoencoder.train()
#         total_loss = 0.0
#         for idx, x in enumerate(data):
#             x = x.to(device) 
#             opt.zero_grad()
#             x_hat = autoencoder(x,idx)
#             MSELoss = ((x - x_hat)**2).sum() 
#             loss =  ((x - x_hat)**2).sum()  + autoencoder.encoder.kl 
#             total_loss = loss
#             total = (x**2).sum()
#             loss.backward()
#             opt.step()
#         train_losses[epoch] = total_loss
#     return autoencoder, MSELoss, train_losses

# def train_NICA(autoencoder, data, epochs=100):
#     opt = torch.optim.Adam(autoencoder.parameters())
#     recon_output = torch.zeros(12600,96) #######
#     train_losses = torch.zeros(epochs,1)
#     for epoch in range(epochs):
#         autoencoder.train()
#         total_loss = 0.0
#         for idx, x in enumerate(data):
#             x = x.to(device) 
#             opt.zero_grad()
#             x_hat = autoencoder(x,idx)
#             recon_output[idx] = x_hat
#             MSELoss = ((x - x_hat)**2).sum() 
#             loss = MSELoss + autoencoder.encoder.kl + autoencoder.encoder.mi
#             total = (x*x).sum()
#             total_loss = loss
#             loss.backward()
#             opt.step()
#         train_losses[epoch] = total_loss
#     return autoencoder, MSELoss, loss, total, recon_output, train_losses

# ######## train with soft constraint #######

# def train_2d(autoencoder, data, latent, k, epochs=200):
#     opt = torch.optim.Adam(autoencoder.parameters())
#     for epoch in range(epochs):
#         for idx, x in enumerate(data):
#             x = x.to(device) 
#             opt.zero_grad()
#             z = autoencoder.encoder(x)
#             x_hat = autoencoder(x)
#             loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl + k * ((z[0] - latent[idx])**2).sum()  # L2 NORM (learned, latent one) 
#             loss.backward()
#             opt.step()
#     return autoencoder, loss

# # def CalcMI(z1,z2,z3,z4):
# #     z1 = z1.to('cpu').detach().numpy
# #     z2 = z2.to('cpu').detach().numpy
# #     z3 = z3.to('cpu').detach().numpy
# #     z3 = z4.to('cpu').detach().numpy
# #     MI = mutual_info_regression(z1, z3) + mutual_info_regression(z1, z4) + mutual_info_regression(z2, z3) + mutual_info_regression(z2, z4) + mutual_info_regression(z3, z4)
# #     MI = MI[0]
# #     return MI

# def train_basic(autoencoder, data, max_epochs):
#     opt = torch.optim.Adam(autoencoder.parameters())
#     for epoch in range(max_epochs):
#         train_loss, start = 0.0, time.time()
#         for x in tqdm(data,desc=f"Epoch {epoch + 1}/{max_epochs}"):
#             x = x.to(device) 
#             opt.zero_grad()
#             z = autoencoder.encoder(x)
#             x_hat = autoencoder(x)
#             MSEloss = ((x - x_hat)**2).sum()
#             loss = MSEloss + autoencoder.encoder.kl
#             train_loss += loss.item()
#             loss.backward()
#             opt.step()
#     return autoencoder

########### training ###########
latent_dims = 2
class_size = 2
feature_size = 96
model = CVAE(feature_size, latent_dims, class_size).to(device)
max_epochs = 5

optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

max_epochs = 1
losses= []
NRMSE = []


# print(net2)

# opt = torch.optim.Adam(net2.parameters())
# losses = []

# print("training on", device)

# for epoch in range(max_epochs):
#     train_loss, start = 0.0, time.time()
#     for x in tqdm(Phi_rt_feature,desc=f"Epoch {epoch + 1}/{max_epochs}"):
#         opt.zero_grad()
#         x = x.to(device)
#         x_hat = net2(x)
#         z = net2.encoder(x)
#         MSELoss = ((x - x_hat)**2).sum()
#         loss = MSELoss + net2.encoder.kl
#         loss.backward()
#         opt.step()
#         train_loss += loss.item()
#     # latent = net(Phi_rt_feature)
#     # mi = LatentMutualInfo(latent[0],latent[1])
#     # mutualinfo.append(mi)
#     train_loss /= Phi_rt_feature.shape[0]
#     losses.append(train_loss)
#     print('epoch %d, train loss %.4f , time %.1f sec'
#           % (epoch+1, train_loss, time.time() - start))
    
# plt.figure()
# plt.plot(losses, label='Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss Curve')
# plt.legend()
# plt.savefig(os.path.join('./assets/graphs', 'loss_net2.jpg'))

# print("Training complete")

# recon_output = net2(Phi_rt_feature)
# recon_output = recon_output.to('cpu').detach().numpy()
# np.save('./assets/stored_parameters/recon_output_net2.npy', recon_output)
# latent_space = net2.encoder(Phi_rt_feature)
# latent_space = latent_space.to('cpu').detach().numpy()
# np.save('./assets/stored_parameters/latent_space_net2.npy', latent_space)


# net4 = VAE(latent_dims=4).to(device)
# print(net4)

# opt = torch.optim.Adam(net4.parameters())
# losses_net4 = []

# print("training on", device)

# for epoch in range(max_epochs):
#     train_loss, start = 0.0, time.time()
#     for x in tqdm(Phi_rt_feature,desc=f"Epoch {epoch + 1}/{max_epochs}"):
#         opt.zero_grad()
#         x = x.to(device)
#         x_hat = net4(x)
#         z = net4.encoder(x)
#         MSELoss = ((x - x_hat)**2).sum()
#         loss = MSELoss + net4.encoder.kl
#         loss.backward()
#         opt.step()
#         train_loss += loss.item()
#     # latent = net(Phi_rt_feature)
#     # mi = LatentMutualInfo(latent[0],latent[1])
#     # mutualinfo.append(mi)
#     train_loss /= Phi_rt_feature.shape[0]
#     losses_net4.append(train_loss)
#     print('epoch %d, train loss %.4f , time %.1f sec'
#           % (epoch+1, train_loss, time.time() - start))
    
# plt.figure()
# plt.plot(losses_net4, label='Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss Curve')
# plt.legend()
# plt.savefig(os.path.join('./assets/graphs', 'loss_net4.jpg'))

# print("Training complete")

# recon_output = net4(Phi_rt_feature)
# recon_output = recon_output.to('cpu').detach().numpy()
# np.save('./assets/stored_parameters/recon_output_net4.npy', recon_output)
# latent_space = net4.encoder(Phi_rt_feature)
# latent_space = latent_space.to('cpu').detach().numpy()
# np.save('./assets/stored_parameters/latent_space_net4.npy', latent_space)

print("training on", device)


for epoch in range(max_epochs):
    train_loss, nrmse_, start = 0.0, 0.0, time.time()
    for idx, (x,t1t2) in tqdm(enumerate(train_dataset),desc=f"Epoch {epoch + 1}/{max_epochs}"):
    # tqdm(train_dataset,desc=f"Epoch {epoch + 1}/{max_epochs}"):
        x, t1t2 = x.to(device), t1t2.to(device)
        x = x.unsqueeze(0)
        t1t2 = t1t2.unsqueeze(0)
        recon_x, mu, logvar = model(x, t1t2)
        optimizer.zero_grad()
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()
        rmse = ((recon_x - x)**2).sum()
        total = (x**2).sum()
        train_loss += loss.detach().cpu().numpy()
        optimizer.step()
    train_loss /= Phi_rt_feature.shape[0]
    nrmse_ = torch.sqrt(rmse / total)
    NRMSE.append(nrmse_)
    losses.append(train_loss)
    print('epoch %d, train loss %.4f , NRMSE %.4f , time %.1f sec'
          % (epoch+1, train_loss, nrmse_, time.time() - start))

plt.figure()
plt.plot(losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
# plt.savefig(os.path.join('./lossGraphs', 'losses_test.jpg'))

print("Training complete")

recon_output, latent_space_mu, latent_space_logvar  = model(Phi_rt_feature, t1t2_normalized)
recon_output = recon_output.to('cpu').detach().numpy()
np.save('./assets/recon_output_test.npy', recon_output)
latent_space = model.reparameterize(latent_space_mu,latent_space_logvar)
latent_space = latent_space.to('cpu').detach().numpy()
np.save('./assets/latent_space_test.npy', latent_space)




# vae1d.load_state_dict(torch.load("vae1d_weights.pth"))
# print("vae1d state_dict:")
# for param_tensor in vae1d.state_dict():
#     print(param_tensor, "\t",vae1d.state_dict()[param_tensor].size())

# vae2d = VariationalAutoencoder(latent_dims=2).to(device)
# vae4d_hc = VariationalAutoencoder_hc(latent_dims=4).to(device)
# vae4d_hc_m = VariationalAutoencoder_hc_m(latent_dims=4).to(device)
# vae_fixed_linear = VariationalAutoencoder_linear4(latent_dims=4).to(device)
# vae_fixed_nonlinear = VariationalAutoencoder_nonlinear4(latent_dims=4).to(device)
# latent_1 = np.load('latent_1.npy'
# latent_1 = torch.from_numpy(latent_1)


# print(data.shape)
# print(latent_1.shape)

# vae2d, loss2d = train_2d(vae2d, data, latent_1, k=1)
# latent_2 = vae2d.encoder(data)
# latent_2 = latent_2.to('cpu').detach().numpy()
# np.save('latent_2_k_1_5.npy', latent_2)
# print(loss2d.item())

# vae_r2, loss_r2 = train_2d(vae2d, data, ridx_mu, k = 1)
# latent_r2 = vae_r2.encoder(data)
# latent_r2 = latent_r2.to('cpu').detach().numpy()
# np.save('latent_r2_2.npy', latent_r2)

# vae4d = VAE(latent_dims=4).to(device)

# vae4d, loss4d = train_4d(vae4d, Phi_rt_feature)
# z = vae4d.encoder(data)
# print(z.shape)
# print(loss4d.item())
# print(total.item())
# latent_4d = vae4d.encoder(data)
# latent_4d = latent_4d.to('cpu').detach().numpy()


# vae4d_hc, loss4d_hc, MSEloss, total = train(vae4d_hc,Phi_rt_feature)
# percentage = MSEloss/total
# print(loss4d_hc.item())
# print(percentage.item())
# latent_4d = torch.zeros(12600, 4)
# phi_rt_feature_hat = torch.zeros(12600, 96)
# for idx, x in enumerate(Phi_rt_feature):
#    z, latent_4d[idx], log_var = vae4d_hc.encoder(x,idx)
#    phi_rt_feature_hat[idx] = vae4d_hc(x,idx)
# phi_rt_feature_hat = phi_rt_feature_hat.to('cpu').detach().numpy
# np.save('Phi_rt_feature_hat_linear.npy', phi_rt_feature_hat)
# latent_4d = latent_4d.to('cpu').detach().numpy()
# np.save('latent_4d_hard_constraint_f.npy', latent_4d)

# vae4d_hc, loss4d_hc, MSEloss, total = train(vae4d_hc,data)
# percentage = MSEloss/total
# print(loss4d_hc.item())
# print(percentage.item())
# latent_4d = torch.zeros(12600, 4)
# phi_rt_hat = torch.zeros(12600, 96)
# for idx, x in enumerate(data):
#    z, latent_4d[idx], log_var = vae4d_hc.encoder(x,idx)
#    phi_rt_hat[idx] = vae4d_hc(x,idx)
# phi_rt_hat = phi_rt_hat.to('cpu').detach().numpy
# np.save('Phi_rt_hat_linear.npy', phi_rt_hat)
# latent_4d = latent_4d.to('cpu').detach().numpy()
# np.save('latent_4d_hard_constraint.npy', latent_4d)



# vae4d_hc_m, loss4d_hc_m, total, Phi_rt_feature_m_hat= train(vae4d_hc_m, Phi_rt_feature)
# print(loss4d_hc_m.item())
# print(total.item())
# torch.save(vae4d_hc_m.state_dict(), 'latent_4d_hard_constraint_modified_f.pth')
# torch.save(Phi_rt_feature_m_hat, 'Phi_rt_feature_m_hat.pth')
# latent_4d_m = torch.zeros(12600, 4)
# for idx, x in enumerate(Phi_rt_feature):
#    z, latent_4d_m[idx], log_var = vae4d_hc_m.encoder(x,idx)
# latent_4d_m = latent_4d_m.to('cpu').detach().numpy()
# np.save('latent_4d_hard_constraint_modified_f.npy', latent_4d_m)

# vae4d_hc_m, loss4d_hc_m, total, Phi_rt_init_m_hat= train(vae4d_hc_m, data)
# print(loss4d_hc_m.item())
# print(total.item())
# torch.save(vae4d_hc_m.state_dict(), 'latent_4d_hard_constraint_modified.pth')
# torch.save(Phi_rt_init_m_hat, 'Phi_rt_init_m_hat.pth')
# latent_4d_m = torch.zeros(12600, 4)
# for idx, x in enumerate(data):
#    z, latent_4d_m[idx], log_var = vae4d_hc_m.encoder(x,idx)
# latent_4d_m = latent_4d_m.to('cpu').detach().numpy()
# np.save('latent_4d_hard_constraint_modified.npy', latent_4d_m)

# vae4d_hc, loss4d_hc, total, Phi_rt_init_hat= train(vae4d_hc, data)
# print(loss4d_hc.item())
# print(total.item())
# torch.save(vae4d_hc.state_dict(), 'latent_4d_hard_constraint_linear.pth')
# torch.save(Phi_rt_init_hat, 'Phi_rt_init_hat.pth')
# latent_4d = torch.zeros(12600, 4)
# for idx, x in enumerate(data):
#    z, latent_4d[idx], log_var = vae4d_hc.encoder(x,idx)
# latent_4d = latent_4d.to('cpu').detach().numpy()
# np.save('latent_4d_hard_constraint_linear.npy', latent_4d)

# vae4d_hc, loss4d_hc, total, Phi_rt_feature_hat= train(vae4d_hc, Phi_rt_feature)
# print(loss4d_hc.item())
# print(total.item())
# torch.save(vae4d_hc.state_dict(), 'latent_4d_hard_constraint_linear_f.pth')
# torch.save(Phi_rt_feature_hat, 'Phi_rt_feature_hat.pth')
# latent_4d = torch.zeros(12600, 4)
# for idx, x in enumerate(Phi_rt_feature):
#    z, latent_4d[idx], log_var = vae4d_hc.encoder(x,idx)
# latent_4d = latent_4d.to('cpu').detach().numpy()
# np.save('latent_4d_hard_constraint_linear_f.npy', latent_4d)

# vae_fixed_nonlinear_feature, loss_linear_feature, total, Phi_rt_feature_hat, train_losses= train(vae_fixed_nonlinear, data)
# train_losses = train_losses.to('cpu').detach().numpy()
# np.save('training_curve_nonlinear_init.npy', train_losses)
# vae_fixed_linear_feature, loss_linear_feature, total, Phi_rt_feature_hat, train_losses= train(vae_fixed_linear, data)
# train_losses = train_losses.to('cpu').detach().numpy()
# np.save('training_curve_linear_init.npy', train_losses)
# plt.plot(train_losses, label='Total Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()
# print(loss_linear_feature.item())
# print(total.item())
# # torch.save(vae4d_hc.state_dict(), 'latent_4d_fixed_linear.pth')
# torch.save(Phi_rt_feature_hat, 'Phi_rt_feature_hat_fixed_linear.pth')
# # latent_4d = torch.zeros(12600, 4)
# # for idx, x in enumerate(data):
# #    z, latent_4d[idx], log_var = vae4d_hc.encoder(x,idx)
# # latent_4d = latent_4d.to('cpu').detach().numpy()
# # np.save('latent_4d_hard_constraint_linear.npy', latent_4d)

# vae_fixed_linear_feature, loss_linear_feature, total, Phi_rt_feature_hat= train(vae_fixed_linear, Phi_rt_feature)
# print(loss_linear_feature.item())
# print(total.item())
# torch.save(Phi_rt_feature_hat, 'Phi_rt_feature_hat_fixed_nonlinear.pth')
# latent_4d = torch.zeros(12600, 4)
# for idx, x in enumerate(data):
#    z, latent_4d[idx], log_var = vae_fixed_linear.encoder(x,idx)
# latent_4d = latent_4d.to('cpu').detach().numpy()
# np.save('latent_fixed_linear.npy', latent_4d)



# vae_fixed_linear, loss_linear, total, Phi_rt_init_hat= train(vae_fixed_linear, data)
# print(loss_linear.item())
# print(total.item())
# torch.save(vae4d_hc.state_dict(), 'latent_4d_fixed_linear.pth')
# torch.save(Phi_rt_init_hat, 'Phi_rt_init_hat_fixed_linear.pth')
# # latent_4d = torch.zeros(12600, 4)
# # for idx, x in enumerate(data):
# #    z, latent_4d[idx], log_var = vae4d_hc.encoder(x,idx)
# # latent_4d = latent_4d.to('cpu').detach().numpy()
# # np.save('latent_4d_hard_constraint_linear.npy', latent_4d)

# vae_fixed_nonlinear, loss_nonlinear, total, Phi_rt_init_hat= train(vae_fixed_linear, data)
# print(loss_nonlinear.item())
# print(total.item())
# torch.save(vae4d_hc.state_dict(), 'latent_4d_fixed_nonlinear.pth')
# torch.save(Phi_rt_init_hat, 'Phi_rt_init_hat_fixed_nonlinear.pth')


# train_losses = []  # Total loss
# reconstruction_losses = []  # Reconstruction loss
# kl_losses = []  # KL divergence loss

# optimizer = optim.Adam(vae_fixed_linear.parameters(), lr=0.001)

# num_epochs = 20

# for epoch in range(num_epochs):
#     vae_fixed_linear.train()
#     total_loss = 0.0
#     # total_reconstruction_loss = 0.0
#     # total_kl_loss = 0.0

#     for idx, x in enumerate(Phi_rt_feature):
#         optimizer.zero_grad()

#         # Forward pass
#         recon_x = vae_fixed_linear(x,idx)

#         # Compute the reconstruction loss and KL divergence
#         reconstruction_loss = (recon_x - x)**2
#         # kl_loss = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())
#         loss = reconstruction_loss #+ kl_loss

#         # Backpropagation and optimization
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         # total_reconstruction_loss += reconstruction_loss.item()
#         # total_kl_loss += kl_loss.item()

#     # Calculate average loss for this epoch
#     avg_loss = total_loss / 12600
#     # avg_reconstruction_loss = total_reconstruction_loss / 12600
#     # avg_kl_loss = total_kl_loss / 12600

#     # Store the losses in their respective lists
#     train_losses.append(avg_loss)
#     # reconstruction_losses.append(avg_reconstruction_loss)
#     # kl_losses.append(avg_kl_loss)

#     print(f"Epoch {epoch + 1}/{num_epochs}:")
#     print(f"Total Loss: {avg_loss:.4f}")

# plt.figure(figsize=(12, 4))
# # plt.subplot(1, 3, 1)
# plt.plot(train_losses, label='Total Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# # plt.subplot(1, 3, 2)
# # plt.plot(reconstruction_losses, label='Reconstruction Loss')
# # plt.xlabel('Epoch')
# # plt.ylabel('Loss')
# # plt.legend()

# # plt.subplot(1, 3, 3)
# # plt.plot(kl_losses, label='KL Divergence')
# # plt.xlabel('Epoch')
# # plt.ylabel('Loss')
# # plt.legend()

# plt.show()

# vae_test = vae_test6(latent_dims=4).to(device)

# vae_test, mseloss, total= train_test(vae_test, Phi_rt_feature)
# print(mseloss.item())
# print(total.item())

# vae_test = vae_test5(latent_dims=4).to(device)
# vae_test, mse, train_losses= train_test_short(vae_test, Phi_rt_feature)
# print(mse.item())
# train_losses= train_losses.to('cpu').detach().numpy()
# np.save('train_losses_6.npy', train_losses)
# # torch.save(vae_test.state_dict(), 'test_6.pth')



