import os
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torch.optim as optim
from vae import VariationalAutoencoder
from vae_constraint import VariationalAutoencoder_hc, VariationalAutoencoder_hc_m
from model import VAE
import torchvision
import numpy as np
import scipy
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from torch.utils.tensorboard import SummaryWriter

#Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#data
data_file = os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'realtime_GT_imgs.mat')
mat = scipy.io.loadmat(data_file)
data = mat['Phi_rt_init'] 
data_real = data.real
data_imag = data.imag
data_con = np.concatenate((data_real, data_imag))
data_con = data_con.transpose()
data = torch.from_numpy(data_con).float()

#data_feature
data_file= os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'feature.mat')
mat = scipy.io.loadmat(data_file)
Phi_rt_feature = mat['Phi_rt_feature'] 
Phi_rt_feature_real = Phi_rt_feature.real
Phi_rt_feature_imag = Phi_rt_feature.imag
Phi_rt_feature_con = np.concatenate((Phi_rt_feature_real, Phi_rt_feature_imag))
Phi_rt_feature_con = Phi_rt_feature_con.transpose()
Phi_rt_feature = torch.from_numpy(Phi_rt_feature_con).float()

# print(data_con.shape)

seg_file = os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'TI.mat')
segidx = scipy.io.loadmat(seg_file)
segidx = segidx['Segidx']

binning_file = os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'binning_GT.mat')
binning_file = scipy.io.loadmat(binning_file)
hidx = binning_file['Hidx'] # Cardiac
ridx = binning_file['Ridx'] # Respiratory


ridx_mu = (ridx - ridx.mean())/ridx.std()
segidx_mu = (segidx - segidx.mean())/segidx.std()

# create ramp matrix
array_1 = np.ones(360)
array_2 = 2 * array_1
array_3 = 3 * array_1
array_4 = 4 * array_1
array_5 = 5 * array_1
array = np.concatenate((array_1, array_2, array_3),axis=0)
array = np.concatenate((array, array_4, array_5),axis=0)
array = np.tile(array, 7)

ramp_latent_mu = (array-array.mean())/array.std()

ramp_latent_mu = torch.from_numpy(ramp_latent_mu)
segidx_mu  = torch.from_numpy(segidx_mu)


def train(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        for idx, x in enumerate(data):
            x = x.to(device) 
            opt.zero_grad()
            x_hat = autoencoder(x,idx)
            loss =  ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            MSELoss = ((x - x_hat)**2).sum()  
            total = (x*x).sum()
            loss.backward()
            opt.step()
    return autoencoder, loss, MSELoss, total

def train_2d(autoencoder, data, latent, k, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        for idx, x in enumerate(data):
            x = x.to(device) 
            opt.zero_grad()
            z = autoencoder.encoder(x)
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl + k * ((z[0] - latent[idx])**2).sum()  # L2 NORM (learned, latent one) 
            loss.backward()
            opt.step()
    return autoencoder, loss

def train_4d(autoencoder, data, para1, para2, m, n, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        for idx, x in enumerate(data):
            x = x.to(device) 
            opt.zero_grad()
            z = autoencoder.encoder(x)
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl + m * ((z[0] - para1[idx])**2).sum() + n * ((z[1] - para2[idx])**2).sum()
            loss.backward()
            opt.step()
    return autoencoder, loss

# vae1d = VariationalAutoencoder(latent_dims=1).to(device)
# vae1d, loss1d = train(vae1d, data)
# latent_1 = vae1d.encoder(data)
# latent_1 = latent_1.to('cpu').detach().numpy()
# np.save('latent_1.npy', latent_1)
# print(loss1d.item())

# vae1d.load_state_dict(torch.load("vae1d_weights.pth"))
# print("vae1d state_dict:")
# for param_tensor in vae1d.state_dict():
#     print(param_tensor, "\t",vae1d.state_dict()[param_tensor].size())

# vae2d = VariationalAutoencoder(latent_dims=2).to(device)
# vae4d_hc = VariationalAutoencoder_hc(latent_dims=4).to(device)
vae4d_hc_m = VariationalAutoencoder_hc_m(latent_dims=4).to(device)
# latent_1 = np.load('latent_1.npy')
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

# vae4d, loss4d = train_4d(vae4d, data, segidx_mu, ramp_latent_mu, 400, 400)
# latent_4d = vae4d.encoder(data)
# latent_4d = latent_4d.to('cpu').detach().numpy()
# np.save('latent_4d_400.npy', latent_4d)

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



vae4d_hc_m, loss4d_hc_m, MSELoss, total= train(vae4d_hc_m, data)
print(loss4d_hc_m.item())
print(MSELoss.item())
print(total.item())
latent_4d_m = torch.zeros(12600, 4)
for idx, x in enumerate(data):
   z, latent_4d_m[idx], log_var = vae4d_hc_m.encoder(x,idx)
latent_4d_m = latent_4d_m.to('cpu').detach().numpy()
np.save('latent_4d_hard_constraint_modified.npy', latent_4d_m)







# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/experiment_1')
# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)