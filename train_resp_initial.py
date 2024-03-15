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
from models import vae_NT1NT2Phasecard_resp, vae_LT1NT2Phasecard_resp, vae_LT1LT2Phasecard_resp, vae_NT1LT2Phasecard_resp
from models import vae_NT1NT2Sincard_resp, vae_LT1NT2Sincard_resp, vae_LT1LT2Sincard_resp, vae_NT1LT2Sincard_resp
from models import vae_NT1NT2Coscard_resp, vae_LT1NT2Coscard_resp, vae_LT1LT2Coscard_resp, vae_NT1LT2Coscard_resp
from models import vae_NT1NT2CosSincard_resp, vae_LT1NT2CosSincard_resp, vae_LT1LT2CosSincard_resp, vae_NT1LT2CosSincard_resp
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

def train(vae, data, epochs=100):
    opt = torch.optim.Adam(vae.parameters())
    # recon_output = torch.zeros(12600,96) 
    # train_losses = torch.zeros(epochs,1)
    for epoch in range(epochs):
        # total_loss = 0.0
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
    return vae, MSELoss #, recon_output

# vae_NT1_NT2_Phase_resp = vae_NT1NT2Phasecard_resp(latent_dims=4).to(device)
# vae_NT1_NT2_Phase_resp.load_state_dict(torch.load('NT1_NT2_Phasecard.pth'), strict=False)
# vae_NT1_NT2_Phase_resp, MSELoss, Phi_rt_feature_hat = train(vae_NT1_NT2_Phase_resp, Phi_rt_feature)
# torch.save(vae_NT1_NT2_Phase_resp.state_dict(), 'NT1_NT2_Phase_resp.pth')
# torch.save(Phi_rt_feature_hat, 'output_NT1_NT2_Phase_resp.pth')
# print(MSELoss.item)
# latent_variables = torch.zeros(12600, 4)
# for idx, x in enumerate(data):
#    z, latent_variables[idx], log_var = vae_NT1_NT2_Phase_resp.encoder(x,idx)
# latent_variables= latent_variables.to('cpu').detach().numpy()
# np.save('latent_NT1_NT2_Phase.npy', latent_variables)

# vae_LT1_NT2_Phase_resp = vae_LT1NT2Phasecard_resp(latent_dims=4).to(device)
# vae_LT1_NT2_Phase_resp.load_state_dict(torch.load('LT1_NT2_Phasecard.pth'), strict=False)
# vae_LT1_NT2_Phase_resp, MSELoss, Phi_rt_feature_hat = train(vae_LT1_NT2_Phase_resp, Phi_rt_feature)
# torch.save(vae_LT1_NT2_Phase_resp.state_dict(), 'LT1_NT2_Phase_resp.pth')
# torch.save(Phi_rt_feature_hat, 'output_LT1_NT2_Phase_resp.pth')
# latent_variables = torch.zeros(12600, 4)
# for idx, x in enumerate(data):
#    z, latent_variables[idx], log_var = vae_LT1_NT2_Phase_resp.encoder(x,idx)
# latent_variables= latent_variables.to('cpu').detach().numpy()
# np.save('latent_LT1_NT2_Phase.npy', latent_variables)

# vae_NT1_LT2_Phase_resp = vae_NT1LT2Phasecard_resp(latent_dims=4).to(device)
# vae_NT1_LT2_Phase_resp.load_state_dict(torch.load('NT1_LT2_Phasecard.pth'), strict=False)
# vae_NT1_LT2_Phase_resp, MSELoss, Phi_rt_feature_hat = train(vae_NT1_LT2_Phase_resp, Phi_rt_feature)
# torch.save(vae_NT1_LT2_Phase_resp.state_dict(), 'NT1_LT2_Phase_resp.pth')
# torch.save(Phi_rt_feature_hat, 'output_NT1_LT2_Phase_resp.pth')
# # print(MSELoss.item)
# latent_variables = torch.zeros(12600, 4)
# for idx, x in enumerate(data):
#    z, latent_variables[idx], log_var = vae_NT1_LT2_Phase_resp.encoder(x,idx)
# latent_variables= latent_variables.to('cpu').detach().numpy()
# np.save('latent_NT1_LT2_Phase.npy', latent_variables)

vae_NT1_NT2_Sin_resp = vae_NT1NT2Sincard_resp(latent_dims=4).to(device)
# vae_NT1_NT2_Sin_resp.load_state_dict(torch.load('NT1_NT2_Sincard.pth'), strict=False)
# vae_NT1_NT2_Sin_resp, MSELoss = train(vae_NT1_NT2_Sin_resp, Phi_rt_feature)
# torch.save(vae_NT1_NT2_Sin_resp.state_dict(), 'NT1_NT2_Sin_resp.pth')
# torch.save(Phi_rt_feature_hat, 'output_NT1_NT2_Sin_resp.pth')
vae_NT1_NT2_Sin_resp.load_state_dict(torch.load('NT1_NT2_Sin_resp.pth'))
# print(MSELoss.item)
latent_variables = torch.zeros(12600, 4)
for idx, x in enumerate(Phi_rt_feature):
   z, latent_variables[idx], log_var = vae_NT1_NT2_Sin_resp.encoder(x,idx)
latent_variables= latent_variables.to('cpu').detach().numpy()
np.save('latent_NT1_NT2_Sin.npy', latent_variables)

vae_LT1_NT2_Sin_resp = vae_LT1NT2Sincard_resp(latent_dims=4).to(device)
# vae_LT1_NT2_Sin_resp.load_state_dict(torch.load('LT1_NT2_Sincard.pth'), strict=False)
# vae_LT1_NT2_Sin_resp, MSELoss = train(vae_LT1_NT2_Sin_resp, Phi_rt_feature)
# torch.save(vae_LT1_NT2_Sin_resp.state_dict(), 'LT1_NT2_Sin_resp.pth')
vae_LT1_NT2_Sin_resp.load_state_dict(torch.load('LT1_NT2_Sin_resp.pth'))
# torch.save(Phi_rt_feature_hat, 'output_LT1_NT2_Sin_resp.pth')
# print(MSELoss.item)
latent_variables = torch.zeros(12600, 4)
for idx, x in enumerate(Phi_rt_feature):
   z, latent_variables[idx], log_var = vae_LT1_NT2_Sin_resp.encoder(x,idx)
latent_variables= latent_variables.to('cpu').detach().numpy()
np.save('latent_LT1_NT2_Sin.npy', latent_variables)

vae_NT1_LT2_Sin_resp = vae_NT1LT2Sincard_resp(latent_dims=4).to(device)
# vae_NT1_LT2_Sin_resp.load_state_dict(torch.load('NT1_LT2_Sincard.pth'), strict=False)
# vae_NT1_LT2_Sin_resp, MSELoss = train(vae_NT1_LT2_Sin_resp, Phi_rt_feature)
# torch.save(vae_NT1_LT2_Sin_resp.state_dict(), 'NT1_LT2_Sin_resp.pth')
# torch.save(Phi_rt_feature_hat, 'output_NT1_LT2_Sin_resp.pth')
# print(MSELoss.item)
vae_NT1_LT2_Sin_resp.load_state_dict(torch.load('NT1_LT2_Sin_resp.pth'))
latent_variables = torch.zeros(12600, 4)
for idx, x in enumerate(Phi_rt_feature):
   z, latent_variables[idx], log_var = vae_NT1_LT2_Sin_resp.encoder(x,idx)
latent_variables= latent_variables.to('cpu').detach().numpy()
np.save('latent_NT1_LT2_Sin.npy', latent_variables)

vae_LT1_LT2_Sin_resp = vae_LT1LT2Sincard_resp(latent_dims=4).to(device)
# vae_LT1_LT2_Sin_resp.load_state_dict(torch.load('LT1_LT2_Sincard.pth'), strict=False)
# vae_LT1_LT2_Sin_resp, MSELoss = train(vae_LT1_LT2_Sin_resp, Phi_rt_feature)
# torch.save(vae_LT1_LT2_Sin_resp.state_dict(), 'LT1_LT2_Sin_resp.pth')
# # torch.save(Phi_rt_feature_hat, 'output_LT1_LT2_Sin_resp.pth')
# print(MSELoss.item)
vae_LT1_LT2_Sin_resp.load_state_dict(torch.load('LT1_LT2_Sin_resp.pth'))
latent_variables = torch.zeros(12600, 4)
for idx, x in enumerate(Phi_rt_feature):
   z, latent_variables[idx], log_var = vae_LT1_LT2_Sin_resp.encoder(x,idx)
latent_variables= latent_variables.to('cpu').detach().numpy()
np.save('latent_LT1_LT2_Sin.npy', latent_variables)

# vae_NT1_NT2_Cos_resp = vae_NT1NT2Coscard_resp(latent_dims=4).to(device)
# vae_NT1_NT2_Cos_resp.load_state_dict(torch.load('NT1_NT2_Coscard.pth'), strict=False)
# vae_NT1_NT2_Cos_resp, MSELoss = train(vae_NT1_NT2_Cos_resp, Phi_rt_feature)
# torch.save(vae_NT1_NT2_Cos_resp.state_dict(), 'NT1_NT2_Cos_resp.pth')
# # torch.save(Phi_rt_feature_hat, 'output_NT1_NT2_Sin_resp.pth')
# # print(MSELoss.item)
# latent_variables = torch.zeros(12600, 4)
# for idx, x in enumerate(data):
#    z, latent_variables[idx], log_var = vae_NT1_NT2_Cos_resp.encoder(x,idx)
# latent_variables= latent_variables.to('cpu').detach().numpy()
# np.save('latent_NT1_NT2_Cos.npy', latent_variables)

# vae_LT1_NT2_Cos_resp = vae_LT1NT2Coscard_resp(latent_dims=4).to(device)
# vae_LT1_NT2_Cos_resp.load_state_dict(torch.load('LT1_NT2_Coscard.pth'), strict=False)
# vae_LT1_NT2_Cos_resp, MSELoss = train(vae_LT1_NT2_Cos_resp, Phi_rt_feature)
# torch.save(vae_LT1_NT2_Cos_resp.state_dict(), 'LT1_NT2_Cos_resp.pth')
# # torch.save(Phi_rt_feature_hat, 'output_LT1_NT2_Sin_resp.pth')
# # print(MSELoss.item)
# latent_variables = torch.zeros(12600, 4)
# for idx, x in enumerate(data):
#    z, latent_variables[idx], log_var = vae_LT1_NT2_Cos_resp.encoder(x,idx)
# latent_variables= latent_variables.to('cpu').detach().numpy()
# np.save('latent_LT1_NT2_Cos.npy', latent_variables)

# vae_NT1_LT2_Cos_resp = vae_NT1LT2Coscard_resp(latent_dims=4).to(device)
# vae_NT1_LT2_Cos_resp.load_state_dict(torch.load('NT1_LT2_Coscard.pth'), strict=False)
# vae_NT1_LT2_Cos_resp, MSELoss = train(vae_NT1_LT2_Cos_resp, Phi_rt_feature)
# torch.save(vae_NT1_LT2_Cos_resp.state_dict(), 'NT1_LT2_Cos_resp.pth')
# # torch.save(Phi_rt_feature_hat, 'output_NT1_LT2_Sin_resp.pth')
# # print(MSELoss.item)
# latent_variables = torch.zeros(12600, 4)
# for idx, x in enumerate(data):
#    z, latent_variables[idx], log_var = vae_NT1_LT2_Cos_resp.encoder(x,idx)
# latent_variables= latent_variables.to('cpu').detach().numpy()
# np.save('latent_NT1_LT2_Cos.npy', latent_variables)

# vae_LT1_LT2_Cos_resp = vae_LT1LT2Coscard_resp(latent_dims=4).to(device)
# vae_LT1_LT2_Cos_resp.load_state_dict(torch.load('LT1_LT2_Coscard.pth'), strict=False)
# vae_LT1_LT2_Cos_resp, MSELoss = train(vae_LT1_LT2_Cos_resp, Phi_rt_feature)
# torch.save(vae_LT1_LT2_Cos_resp.state_dict(), 'LT1_LT2_Cos_resp.pth')
# # torch.save(Phi_rt_feature_hat, 'output_LT1_LT2_Sin_resp.pth')
# # print(MSELoss.item)
# latent_variables = torch.zeros(12600, 4)
# for idx, x in enumerate(data):
#    z, latent_variables[idx], log_var = vae_LT1_LT2_Cos_resp.encoder(x,idx)
# latent_variables= latent_variables.to('cpu').detach().numpy()
# np.save('latent_LT1_LT2_Cos.npy', latent_variables)

# vae_LT1_LT2_Sin_esti_resp = vae_LT1LT2Sincard_resp(latent_dims=4).to(device)
# # vae_LT1_LT2_Sin_esti_resp.load_state_dict(torch.load('LT1_LT2_Sincard_EstResp.pth'), strict=False)
# # vae_LT1_LT2_Sin_esti_resp, MSELoss1 = train(vae_LT1_LT2_Sin_esti_resp, Phi_rt_feature)
# # torch.save(vae_LT1_LT2_Sin_esti_resp.state_dict(), 'LT1_LT2_Sin_esti_resp.pth')
# vae_LT1_LT2_Sin_esti_resp.load_state_dict(torch.load('LT1_LT2_Sincard_EstResp.pth'))
# # torch.save(Phi_rt_feature_hat, 'output_NT1_NT2_Sin_resp.pth')
# latent_variables = torch.zeros(12600, 4)
# for idx, x in enumerate(Phi_rt_feature):
#    z, latent_variables[idx], log_var = vae_LT1_LT2_Sin_esti_resp.encoder(x,idx)
# latent_variables= latent_variables.to('cpu').detach().numpy()
# np.save('latent_LT1_LT2_Sin_esti_resp.npy', latent_variables)

# vae_LT1_LT2_Cos_esti_resp = vae_LT1LT2Coscard_resp(latent_dims=4).to(device)
# # vae_LT1_LT2_Cos_esti_resp.load_state_dict(torch.load('LT1_LT2_Coscard_EstResp.pth'), strict=False)
# # vae_LT1_LT2_Cos_esti_resp, MSELoss2 = train(vae_LT1_LT2_Cos_esti_resp, Phi_rt_feature)
# # torch.save(vae_LT1_LT2_Cos_esti_resp.state_dict(), 'LT1_LT2_Cos_esti_resp.pth')
# vae_LT1_LT2_Cos_esti_resp.load_state_dict(torch.load('LT1_LT2_Coscard_EstResp.pth'))
# # torch.save(Phi_rt_feature_hat, 'output_NT1_NT2_Sin_resp.pth')
# latent_variables = torch.zeros(12600, 4)
# for idx, x in enumerate(Phi_rt_feature):
#    z, latent_variables[idx], log_var = vae_LT1_LT2_Cos_esti_resp.encoder(x,idx)
# latent_variables= latent_variables.to('cpu').detach().numpy()
# np.save('latent_LT1_LT2_Cos_esti_resp.npy', latent_variables)

# vae_NT1_NT2_Phase_esti_resp = vae_NT1NT2Phasecard_resp(latent_dims=4).to(device)
# # vae_NT1_NT2_Phase_esti_resp.load_state_dict(torch.load('NT1_NT2_Phasecard_EstResp.pth'), strict=False)
# # vae_NT1_NT2_Phase_esti_resp, MSELoss3 = train(vae_NT1_NT2_Phase_esti_resp, Phi_rt_feature)
# # torch.save(vae_NT1_NT2_Phase_esti_resp.state_dict(), 'NT1_NT2_Phase_esti_resp.pth')
# vae_NT1_NT2_Phase_esti_resp.load_state_dict(torch.load('NT1_NT2_Phase_esti_resp.pth'))
# # torch.save(Phi_rt_feature_hat, 'output_NT1_NT2_Sin_resp.pth')
# latent_variables = torch.zeros(12600, 4)
# for idx, x in enumerate(Phi_rt_feature):
#    z, latent_variables[idx], log_var = vae_NT1_NT2_Phase_esti_resp.encoder(x,idx)
# latent_variables= latent_variables.to('cpu').detach().numpy()
# np.save('latent_NT1_NT2_Phase_esti_resp.npy', latent_variables)

# vae_NT1_NT2_Cos_esti_resp = vae_NT1NT2Coscard_resp(latent_dims=4).to(device)
# # vae_NT1_NT2_Cos_esti_resp.load_state_dict(torch.load('NT1_NT2_Coscard_EstResp.pth'), strict=False)
# # vae_NT1_NT2_Cos_esti_resp, MSELoss4 = train(vae_NT1_NT2_Cos_esti_resp, Phi_rt_feature)
# # torch.save(vae_NT1_NT2_Cos_esti_resp.state_dict(), 'NT1_NT2_Cos_esti_resp.pth')
# vae_NT1_NT2_Cos_esti_resp.load_state_dict(torch.load('NT1_NT2_Cos_esti_resp.pth'))
# # torch.save(Phi_rt_feature_hat, 'output_NT1_NT2_Sin_resp.pth')
# latent_variables = torch.zeros(12600, 4)
# for idx, x in enumerate(Phi_rt_feature):
#    z, latent_variables[idx], log_var = vae_NT1_NT2_Cos_esti_resp.encoder(x,idx)
# latent_variables= latent_variables.to('cpu').detach().numpy()
# np.save('latent_NT1_NT2_Cos_esti_resp.npy', latent_variables)

# vae_LT1_NT2_Cos_esti_resp = vae_LT1NT2Coscard_resp(latent_dims=4).to(device)
# # vae_LT1_NT2_Cos_esti_resp.load_state_dict(torch.load('LT1_NT2_Coscard_EstResp.pth'), strict=False)
# # vae_LT1_NT2_Cos_esti_resp, MSELoss5 = train(vae_LT1_NT2_Cos_esti_resp, Phi_rt_feature)
# # torch.save(vae_LT1_NT2_Cos_esti_resp.state_dict(), 'LT1_NT2_Cos_esti_resp.pth')
# vae_LT1_NT2_Cos_esti_resp.load_state_dict(torch.load('LT1_NT2_Cos_esti_resp.pth'))
# # torch.save(Phi_rt_feature_hat, 'output_NT1_NT2_Sin_resp.pth')
# latent_variables = torch.zeros(12600, 4)
# for idx, x in enumerate(Phi_rt_feature):
#    z, latent_variables[idx], log_var = vae_LT1_NT2_Cos_esti_resp.encoder(x,idx)
# latent_variables= latent_variables.to('cpu').detach().numpy()
# np.save('latent_LT1_NT2_Cos_esti_resp.npy', latent_variables)

# print(MSELoss1.item())
# print(MSELoss2.item())
# print(MSELoss3.item())
# print(MSELoss4.item())
# print(MSELoss5.item())

