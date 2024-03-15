import os
import sys
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torch.optim as optim
import torchvision
import numpy as np
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from models import vae_LT1LT2Phasecard, vae_LT1NT2Phasecard, vae_NT1LT2Phasecard, vae_NT1NT2Phasecard
from models import vae_LT1LT2Coscard, vae_LT1NT2Coscard, vae_NT1LT2Coscard, vae_NT1NT2Coscard
from models import vae_LT1LT2CosSincard, vae_LT1NT2CosSincard, vae_NT1LT2CosSincard, vae_NT1NT2CosSincard
from models import vae_LT1LT2Coscard_EstResp, vae_LT1LT2Sincard_EstResp, vae_NT1NT2Coscard_EstResp, vae_NT1NT2Phasecard_EstResp, vae_LT1NT2Coscard_EstResp
#Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# MI = MutualInformation(num_bins=256, sigma=0.4, normalize=True).to(device)

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
segidx_mu = (segidx - segidx.mean())/segidx.std()

binning_file = os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'binning_GT.mat')
binning_file = scipy.io.loadmat(binning_file)
hidx = binning_file['Hidx'] # Cardiac
ridx = binning_file['Ridx'] # Respiratory

hidx_mu = (hidx - hidx.mean())/hidx.std()
ridx_mu = (ridx - ridx.mean())/ridx.std()

ridx_mu = torch.from_numpy(ridx_mu)
hidx_mu = torch.from_numpy(hidx_mu)

vae_LT1_LT2_Phasecard = vae_LT1LT2Phasecard(latent_dims=4).to(device)
vae_LT1_NT2_Phasecard = vae_LT1NT2Phasecard(latent_dims=4).to(device)
vae_NT1_NT2_Phasecard = vae_NT1NT2Phasecard(latent_dims=4).to(device)
vae_NT1_LT2_Phasecard = vae_NT1LT2Phasecard(latent_dims=4).to(device)

vae_LT1_LT2_Coscard = vae_LT1LT2Coscard(latent_dims=4).to(device)
vae_LT1_NT2_Coscard = vae_LT1NT2Coscard(latent_dims=4).to(device)
vae_NT1_NT2_Coscard = vae_NT1NT2Coscard(latent_dims=4).to(device)
vae_NT1_LT2_Coscard = vae_NT1LT2Coscard(latent_dims=4).to(device)

vae_LT1_LT2_CosSincard = vae_LT1LT2CosSincard(latent_dims=5).to(device)
vae_LT1_NT2_CosSincard = vae_LT1NT2CosSincard(latent_dims=5).to(device)
vae_NT1_NT2_CosSincard = vae_NT1NT2CosSincard(latent_dims=5).to(device)
vae_NT1_LT2_CosSincard = vae_NT1LT2CosSincard(latent_dims=5).to(device)

vae_LT1_LT2_Sincard_EstResp = vae_LT1LT2Sincard_EstResp(latent_dims=4).to(device)
vae_LT1_LT2_Coscard_EstResp = vae_LT1LT2Coscard_EstResp(latent_dims=4).to(device)
vae_NT1_NT2_Coscard_EstResp = vae_NT1NT2Coscard_EstResp(latent_dims=4).to(device)
vae_LT1_NT2_Coscard_EstResp = vae_LT1NT2Coscard_EstResp(latent_dims=4).to(device)
vae_NT1_NT2_Phasecard_EstResp = vae_NT1NT2Phasecard_EstResp(latent_dims=4).to(device)


def train(vae, data, epochs=100):
    opt = torch.optim.Adam(vae.parameters())
    # recon_output = torch.zeros(12600,96) 
    train_losses = torch.zeros(epochs,1)
    for epoch in range(epochs):
        total_loss = 0.0
        for idx, x in tqdm(enumerate(data)):
            x = x.to(device) 
            opt.zero_grad()
            x_hat = vae(x,idx)
            # recon_output[idx] = x_hat
            loss =  ((x - x_hat)**2).sum()  + vae.encoder.kl
            MSELoss = ((x - x_hat)**2).sum()  
            total = (x*x).sum()
            total_loss = loss
            loss.backward()
            opt.step()
        # train_losses[epoch] = total_loss
    return vae, MSELoss

# vae_LT1_LT2_Phasecard, loss, MSELoss, total, recon_output = train(vae_LT1_LT2_Phasecard, Phi_rt_feature)
# torch.save(vae_LT1_LT2_Phasecard.state_dict(), 'LT1_LT2_Phasecard.pth')
# print(MSELoss.item())
# vae_NT1_LT2_Coscard, MSELoss1 = train(vae_NT1_LT2_Coscard, Phi_rt_feature)
# torch.save(vae_NT1_LT2_Coscard.state_dict(), 'NT1_LT2_Coscard.pth')
# print(MSELoss1.item())

# vae_LT1_NT2_Coscard, MSELoss2 = train(vae_LT1_NT2_Coscard, Phi_rt_feature)
# torch.save(vae_LT1_NT2_Coscard.state_dict(), 'LT1_NT2_Coscard.pth')
# print(MSELoss2.item())

# vae_NT1_NT2_Coscard, MSELoss3 = train(vae_NT1_NT2_Coscard, Phi_rt_feature)
# torch.save(vae_NT1_NT2_Coscard.state_dict(), 'NT1_NT2_Coscard.pth')
# print(MSELoss3.item())

# vae_LT1_LT2_CosSincard, MSELoss4 = train(vae_LT1_LT2_CosSincard, Phi_rt_feature)
# torch.save(vae_LT1_LT2_CosSincard.state_dict(), 'LT1_LT2_CosSincard.pth')
# print(MSELoss4.item())

# vae_NT1_LT2_CosSincard, MSELoss5 = train(vae_NT1_LT2_CosSincard, Phi_rt_feature)
# torch.save(vae_NT1_LT2_CosSincard.state_dict(), 'NT1_LT2_CosSincard.pth')
# print(MSELoss5.item())

# vae_LT1_NT2_CosSincard, MSELoss6 = train(vae_LT1_NT2_CosSincard, Phi_rt_feature)
# torch.save(vae_LT1_NT2_CosSincard.state_dict(), 'LT1_NT2_CosSincard.pth')
# print(MSELoss6.item())

# vae_NT1_NT2_CosSincard, MSELoss7 = train(vae_NT1_NT2_CosSincard, Phi_rt_feature)
# torch.save(vae_NT1_NT2_CosSincard.state_dict(), 'NT1_NT2_CosSincard.pth')
# print(MSELoss7.item())


# vae_LT1_LT2_Sincard_EstResp, MSELoss1 = train(vae_LT1_LT2_Sincard_EstResp, Phi_rt_feature)
# torch.save(vae_LT1_LT2_Sincard_EstResp.state_dict(), 'LT1_LT2_Sincard_EstResp1.pth')
# latent_variables = torch.zeros(12600, 4)
# for idx, x in enumerate(Phi_rt_feature):
#    z, latent_variables[idx], log_var = vae_LT1_LT2_Sincard_EstResp.encoder(x,idx)
# latent_variables= latent_variables.to('cpu').detach().numpy()
# np.save('latent_LT1_LT2_Sin_esti_resp.npy', latent_variables)


# vae_LT1_LT2_Coscard_EstResp, MSELoss2 = train(vae_LT1_LT2_Coscard_EstResp, Phi_rt_feature)
# torch.save(vae_LT1_LT2_Coscard_EstResp.state_dict(), 'LT1_LT2_Coscard_EstResp1.pth')
# latent_variables = torch.zeros(12600, 4)
# for idx, x in enumerate(Phi_rt_feature):
#    z, latent_variables[idx], log_var = vae_LT1_LT2_Coscard_EstResp.encoder(x,idx)
# latent_variables= latent_variables.to('cpu').detach().numpy()
# np.save('latent_LT1_LT2_Cos_esti_resp.npy', latent_variables)


# vae_NT1_NT2_Coscard_EstResp, MSELoss3 = train(vae_NT1_NT2_Coscard_EstResp, Phi_rt_feature)
# torch.save(vae_NT1_NT2_Coscard_EstResp.state_dict(), 'NT1_NT2_Coscard_EstResp1.pth')
# latent_variables = torch.zeros(12600, 4)
# for idx, x in enumerate(Phi_rt_feature):
#    z, latent_variables[idx], log_var = vae_NT1_NT2_Coscard_EstResp.encoder(x,idx)
# latent_variables= latent_variables.to('cpu').detach().numpy()
# np.save('latent_NT1_NT2_Cos_esti_resp.npy', latent_variables)


# vae_NT1_NT2_Phasecard_EstResp, MSELoss4 = train(vae_NT1_NT2_Phasecard_EstResp, Phi_rt_feature)
# torch.save(vae_NT1_NT2_Phasecard_EstResp.state_dict(), 'NT1_NT2_Phasecard_EstResp1.pth')
# latent_variables = torch.zeros(12600, 4)
# for idx, x in enumerate(Phi_rt_feature):
#    z, latent_variables[idx], log_var = vae_NT1_NT2_Phasecard_EstResp.encoder(x,idx)
# latent_variables= latent_variables.to('cpu').detach().numpy()
# np.save('latent_NT1_NT2_Phase_esti_resp.npy', latent_variables)


# vae_LT1_NT2_Coscard_EstResp, MSELoss5 = train(vae_LT1_NT2_Coscard_EstResp, Phi_rt_feature)
# torch.save(vae_LT1_NT2_Coscard_EstResp.state_dict(), 'LT1_NT2_Coscard_EstResp1.pth')
# latent_variables = torch.zeros(12600, 4)
# for idx, x in enumerate(Phi_rt_feature):
#    z, latent_variables[idx], log_var = vae_LT1_NT2_Coscard_EstResp.encoder(x,idx)
# latent_variables= latent_variables.to('cpu').detach().numpy()
# np.save('latent_LT1_NT2_Cos_esti_resp.npy', latent_variables)


# print(MSELoss1.item())
# print(MSELoss2.item())
# print(MSELoss3.item())
# print(MSELoss4.item())
# print(MSELoss5.item())

vae_LT1_LT2_Phasecard.load_state_dict(torch.load('LT1_LT2_Phasecard.pth'))
recon_LT1_LT2_Phase = torch.zeros(12600,96) 
for idx, x in enumerate(Phi_rt_feature):
    recon_LT1_LT2_Phase[idx] = vae_LT1_LT2_Phasecard(x,idx)
recon_LT1_LT2_Phase = recon_LT1_LT2_Phase.to('cpu').detach().numpy()
np.save('recon_LT1_LT2_Phase.npy', recon_LT1_LT2_Phase)

vae_NT1_NT2_Phasecard.load_state_dict(torch.load('NT1_NT2_Phasecard.pth'))
recon_NT1_NT2_Phase = torch.zeros(12600,96) 
for idx, x in enumerate(Phi_rt_feature):
    recon_NT1_NT2_Phase[idx] = vae_NT1_NT2_Phasecard(x,idx)
recon_NT1_NT2_Phase = recon_NT1_NT2_Phase.to('cpu').detach().numpy()
np.save('recon_NT1_NT2_Phase.npy', recon_NT1_NT2_Phase)














