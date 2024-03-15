import os
import torch; 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
import numpy as np
import scipy
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from scipy import signal


device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
hidx_sin = np.sin(hidx*2*np.pi/20)
hidx_cos = np.cos(hidx*2*np.pi/20)

fft_ridx = np.fft.fft(ridx,axis=0)
frequency_ridx = np.fft.fftfreq(ridx.size,d=7e-3)
d = 7e-3
fs = 1/d
# cutoff_freq = 50 / 60  # Cutoff frequency in Hz
# nyquist_freq = 0.5 * fs  # Nyquist frequency
# mask = np.logical_or(frequency_ridx < -cutoff_freq, frequency_ridx > cutoff_freq)
# filtered_ridx_fft = fft_ridx.copy()
# filtered_ridx_fft[mask] = 0
# filtered_ridx = np.fft.ifft(filtered_ridx_fft, axis=0)


def filter_resp(x, cutoff_freq, t):
    x = x.to('cpu').detach().numpy()
    fft_x = np.fft.fft(x, axis=0)
    mask = np.logical_or(frequency_ridx < -cutoff_freq, frequency_ridx > cutoff_freq)
    filtered_x_fft = fft_x.copy()
    filtered_x_fft[mask] = 0
    filtered_x = np.fft.ifft(filtered_x_fft, axis=0)
    filtered_x_norm = filtered_x - filtered_x.mean()/filtered_x.std()
    filtered_x_norm = torch.from_numpy(filtered_x_norm)
    return filtered_x_norm


    


# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# plt.plot(frequency_ridx, np.abs(fft_ridx))
# plt.xlim(-1,1)
# plt.title('Original Signal')
# plt.grid(True)

# plt.subplot(2, 1, 2)
# plt.plot(frequency_ridx, np.abs(filtered_ridx_fft))
# plt.xlim(-1,1)
# plt.title('Filtered Signal')
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# plt.plot(ridx, label='Original Signal')
# plt.title('Original Signal')
# plt.grid(True)

# plt.subplot(2, 1, 2)
# plt.plot(filtered_ridx, label='Filtered Signal')
# plt.title('Filtered Signal')
# plt.grid(True)

# plt.tight_layout()
# plt.show()

Ti = segidx*2*3.56e-3
T1 = 1.2
Tr = 3.56e-3
alpha = 5*np.pi/180
T1_star = 1/(1/T1 - np.log(np.cos(alpha))/Tr)
u1 = 1 - 2 * np.exp(-Ti/T1_star)

# create ramp matrix
array_1 = np.ones(360)
array_2 = 2 * array_1
array_3 = 3 * array_1
array_4 = 4 * array_1
array_5 = 5 * array_1
array = np.concatenate((array_1, array_2, array_3),axis=0)
array = np.concatenate((array, array_4, array_5),axis=0)
wall_clock = np.tile(array, 7)

tau = (wall_clock - 1) * 15e-3
T2 = 50e-3
u2 = np.exp(-tau/T2)

# plt.plot(u1)
# plt.xlim(0,1800)
# plt.show()

wall_clock_mu = (wall_clock-wall_clock.mean())/wall_clock.std()

# filtered_ridx_mu = (filtered_ridx - filtered_ridx.mean())/filtered_ridx.std()

u1_normalized = (u1 - u1.mean())/u1.std()
u2_normalized = (u2 - u2.mean())/u2.std()

u1_normalized = torch.from_numpy(u1_normalized)
u2_normalized = torch.from_numpy(u2_normalized)
wall_clock_mu = torch.from_numpy(wall_clock_mu)
segidx_mu  = torch.from_numpy(segidx_mu)
hidx_mu = torch.from_numpy(hidx_mu)
hidx_sin = torch.from_numpy(hidx_sin)
hidx_cos = torch.from_numpy(hidx_cos)
ridx_mu = torch.from_numpy(ridx_mu)




#phi_rt_init -- encoder (mlp) --- latent_dim --- decoder(mlp) --- phi_hat

class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        sigma = torch.exp(log_var)
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var
    
class EncoderModified(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderModified, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = u1_normalized[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var


class EncoderLT1LT2Phasecard(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderLT1LT2Phasecard, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_mu[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = ridx_mu[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var  

class EncoderNT1LT2Phasecard(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderNT1LT2Phasecard, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = u1_normalized[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_mu[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = ridx_mu[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var
    
class EncoderLT1NT2Phasecard(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderLT1NT2Phasecard, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_mu[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = ridx_mu[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var

class EncoderNT1NT2Phasecard(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderNT1NT2Phasecard, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = u1_normalized[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_mu[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = ridx_mu[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var
    
class EncoderLT1LT2Sincard(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderLT1LT2Sincard, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_sin[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = ridx_mu[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var
    
class EncoderNT1LT2Sincard(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderNT1LT2Sincard, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = u1_normalized[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_sin[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = ridx_mu[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var
    
class EncoderNT1NT2Sincard(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderNT1NT2Sincard, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = u1_normalized[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_sin[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = ridx_mu[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var
    
class EncoderLT1NT2Sincard(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderLT1NT2Sincard, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_sin[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = ridx_mu[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var
    
class EncoderLT1LT2Coscard(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderLT1LT2Coscard, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_cos[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = ridx_mu[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var
    
class EncoderNT1LT2Coscard(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderNT1LT2Coscard, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = u1_normalized[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_cos[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = ridx_mu[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var
    
class EncoderNT1NT2Coscard(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderNT1NT2Coscard, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = u1_normalized[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_cos[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = ridx_mu[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var

class EncoderLT1NT2Coscard(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderLT1NT2Coscard, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_cos[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = ridx_mu[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var

class EncoderLT1LT2CosSincard(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderLT1LT2CosSincard, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_cos[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = hidx_sin[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        mu[4] = ridx_mu[t]
        log_var[4] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var

class EncoderNT1LT2CosSincard(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderNT1LT2CosSincard, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = u1_normalized[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_cos[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = hidx_sin[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        mu[4] = ridx_mu[t]
        log_var[4] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var
    
class EncoderNT1NT2CosSincard(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderNT1NT2CosSincard, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = u1_normalized[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_cos[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = hidx_sin[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        mu[4] = ridx_mu[t]
        log_var[4] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var
    
class EncoderLT1NT2CosSincard(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderLT1NT2CosSincard, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_cos[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = hidx_sin[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        mu[4] = ridx_mu[t]
        log_var[4] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var






class EncoderLT1LT2Phasecard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderLT1LT2Phasecard_resp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_mu[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var  

class EncoderNT1NT2Phasecard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderNT1NT2Phasecard_resp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = u1_normalized[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_mu[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var

class EncoderLT1NT2Phasecard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderLT1NT2Phasecard_resp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_mu[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var

class EncoderNT1LT2Phasecard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderNT1LT2Phasecard_resp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = u1_normalized[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_mu[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var



class EncoderLT1LT2Sincard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderLT1LT2Sincard_resp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_sin[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var  

class EncoderNT1NT2Sincard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderNT1NT2Sincard_resp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = u1_normalized[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_sin[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var

class EncoderLT1NT2Sincard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderLT1NT2Sincard_resp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_sin[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var

class EncoderNT1LT2Sincard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderNT1LT2Sincard_resp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = u1_normalized[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_sin[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var




class EncoderLT1LT2Coscard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderLT1LT2Coscard_resp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_cos[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var  

class EncoderNT1NT2Coscard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderNT1NT2Coscard_resp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = u1_normalized[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_cos[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var

class EncoderLT1NT2Coscard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderLT1NT2Coscard_resp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_cos[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var

class EncoderNT1LT2Coscard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderNT1LT2Coscard_resp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = u1_normalized[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_cos[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var


class EncoderLT1LT2CosSincard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderLT1LT2CosSincard_resp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_cos[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = hidx_sin[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var  

class EncoderNT1NT2CosSincard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderNT1NT2CosSincard_resp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = u1_normalized[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_cos[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = hidx_sin[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var

class EncoderLT1NT2CosSincard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderLT1NT2CosSincard_resp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_cos[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = hidx_sin[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var

class EncoderNT1LT2CosSincard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderNT1LT2CosSincard_resp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        mu[0] = u1_normalized[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_cos[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = hidx_sin[t]
        log_var[3] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var








class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dims,70),
            nn.ReLU(),
            nn.Linear(70,100),
            nn.ReLU(),
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 96)
        )

    def forward(self, z):
        return self.layers(z)

class vae_LT1LT2Phasecard(nn.Module):
    def __init__(self, latent_dims):
        super(vae_LT1LT2Phasecard, self).__init__()
        self.encoder = EncoderLT1LT2Phasecard(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)
    
class vae_NT1LT2Phasecard(nn.Module):
    def __init__(self, latent_dims):
        super(vae_NT1LT2Phasecard, self).__init__()
        self.encoder = EncoderNT1LT2Phasecard(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)
    
class vae_NT1NT2Phasecard(nn.Module):
    def __init__(self, latent_dims):
        super(vae_NT1NT2Phasecard, self).__init__()
        self.encoder = EncoderNT1NT2Phasecard(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)

class vae_LT1NT2Phasecard(nn.Module):
    def __init__(self, latent_dims):
        super(vae_LT1NT2Phasecard, self).__init__()
        self.encoder = EncoderLT1NT2Phasecard(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)

class vae_LT1LT2Sincard(nn.Module):
    def __init__(self, latent_dims):
        super(vae_LT1LT2Sincard, self).__init__()
        self.encoder = EncoderLT1LT2Sincard(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)
    
class vae_LT1NT2Sincard(nn.Module):
    def __init__(self, latent_dims):
        super(vae_LT1NT2Sincard, self).__init__()
        self.encoder = EncoderLT1NT2Sincard(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)
    
class vae_NT1LT2Sincard(nn.Module):
    def __init__(self, latent_dims):
        super(vae_NT1LT2Sincard, self).__init__()
        self.encoder = EncoderNT1LT2Sincard(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)

class vae_NT1NT2Sincard(nn.Module):
    def __init__(self, latent_dims):
        super(vae_NT1NT2Sincard, self).__init__()
        self.encoder = EncoderNT1NT2Sincard(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)
    


class vae_LT1LT2Coscard(nn.Module):
    def __init__(self, latent_dims):
        super(vae_LT1LT2Coscard, self).__init__()
        self.encoder = EncoderLT1LT2Coscard(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)
    
class vae_NT1LT2Coscard(nn.Module):
    def __init__(self, latent_dims):
        super(vae_NT1LT2Coscard, self).__init__()
        self.encoder = EncoderNT1LT2Coscard(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)
    
class vae_NT1NT2Coscard(nn.Module):
    def __init__(self, latent_dims):
        super(vae_NT1NT2Coscard, self).__init__()
        self.encoder = EncoderNT1NT2Coscard(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)

class vae_LT1NT2Coscard(nn.Module):
    def __init__(self, latent_dims):
        super(vae_LT1NT2Coscard, self).__init__()
        self.encoder = EncoderLT1NT2Coscard(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)
    
class vae_LT1LT2CosSincard(nn.Module):
    def __init__(self, latent_dims):
        super(vae_LT1LT2CosSincard, self).__init__()
        self.encoder = EncoderLT1LT2CosSincard(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)
    
class vae_NT1LT2CosSincard(nn.Module):
    def __init__(self, latent_dims):
        super(vae_NT1LT2CosSincard, self).__init__()
        self.encoder = EncoderNT1LT2CosSincard(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)
    
class vae_NT1NT2CosSincard(nn.Module):
    def __init__(self, latent_dims):
        super(vae_NT1NT2CosSincard, self).__init__()
        self.encoder = EncoderNT1NT2CosSincard(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)

class vae_LT1NT2CosSincard(nn.Module):
    def __init__(self, latent_dims):
        super(vae_LT1NT2CosSincard, self).__init__()
        self.encoder = EncoderLT1NT2CosSincard(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)






class vae_LT1LT2Phasecard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_LT1LT2Phasecard_resp, self).__init__()
        self.encoder = EncoderLT1LT2Phasecard_resp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)

class vae_NT1NT2Phasecard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_NT1NT2Phasecard_resp, self).__init__()
        self.encoder = EncoderNT1NT2Phasecard_resp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)

class vae_LT1NT2Phasecard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_LT1NT2Phasecard_resp, self).__init__()
        self.encoder = EncoderLT1NT2Phasecard_resp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)
    
class vae_NT1LT2Phasecard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_NT1LT2Phasecard_resp, self).__init__()
        self.encoder = EncoderNT1LT2Phasecard_resp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)






class vae_LT1LT2Sincard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_LT1LT2Sincard_resp, self).__init__()
        self.encoder = EncoderLT1LT2Sincard_resp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)

class vae_NT1NT2Sincard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_NT1NT2Sincard_resp, self).__init__()
        self.encoder = EncoderNT1NT2Sincard_resp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)

class vae_LT1NT2Sincard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_LT1NT2Sincard_resp, self).__init__()
        self.encoder = EncoderLT1NT2Sincard_resp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)
    
class vae_NT1LT2Sincard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_NT1LT2Sincard_resp, self).__init__()
        self.encoder = EncoderNT1LT2Sincard_resp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)





class vae_LT1LT2Coscard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_LT1LT2Coscard_resp, self).__init__()
        self.encoder = EncoderLT1LT2Coscard_resp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)

class vae_NT1NT2Coscard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_NT1NT2Coscard_resp, self).__init__()
        self.encoder = EncoderNT1NT2Coscard_resp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)

class vae_LT1NT2Coscard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_LT1NT2Coscard_resp, self).__init__()
        self.encoder = EncoderLT1NT2Coscard_resp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)
    
class vae_NT1LT2Coscard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_NT1LT2Coscard_resp, self).__init__()
        self.encoder = EncoderNT1LT2Coscard_resp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)





class vae_LT1LT2CosSincard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_LT1LT2CosSincard_resp, self).__init__()
        self.encoder = EncoderLT1LT2CosSincard_resp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)

class vae_NT1NT2CosSincard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_NT1NT2CosSincard_resp, self).__init__()
        self.encoder = EncoderNT1NT2CosSincard_resp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)

class vae_LT1NT2CosSincard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_LT1NT2CosSincard_resp, self).__init__()
        self.encoder = EncoderLT1NT2CosSincard_resp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)
    
class vae_NT1LT2CosSincard_resp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_NT1LT2CosSincard_resp, self).__init__()
        self.encoder = EncoderNT1LT2CosSincard_resp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)



class EncoderLT1LT2Sincard_EstiResp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderLT1LT2Sincard_EstiResp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def compute_cutoff_frequency(self, t):
        # Linear relationship between time 't' and cutoff frequency
        cutoff_freq = t * self.max_cutoff_freq
        return cutoff_freq

    def low_pass_filter(self, x, cutoff_freq):
        # Implement a low-pass filter based on 'cutoff_freq'
        nyquist_freq = 0.5
        normalized_cutoff_freq = cutoff_freq / nyquist_freq

        # Design a Butterworth low-pass filter
        b, a = signal.butter(4, normalized_cutoff_freq, btype='low')
        filtered_x = signal.lfilter(b, a, x, axis=-1)

        return filtered_x

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        resp = mu[3]
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_sin[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        cutoff_freq = self.compute_cutoff_frequency(t)
        mu[3] = self.low_pass_filter(resp, cutoff_freq)
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var

class vae_LT1LT2Sincard_EstResp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_LT1LT2Sincard_EstResp, self).__init__()
        self.encoder = EncoderLT1LT2Sincard_EstiResp(latent_dims,max_cutoff_freq = 50/60)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)

class EncoderLT1LT2Coscard_EstiResp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderLT1LT2Coscard_EstiResp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        resp = mu[3]
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_cos[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = filter_resp(resp,50/60)
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var

class vae_LT1LT2Coscard_EstResp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_LT1LT2Coscard_EstResp, self).__init__()
        self.encoder = EncoderLT1LT2Coscard_EstiResp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)

class EncoderNT1NT2Phasecard_EstiResp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderNT1NT2Phasecard_EstiResp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        resp = mu[3]
        mu[0] = u1_normalized[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_mu[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = filter_resp(resp,50/60)
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var

class vae_NT1NT2Phasecard_EstResp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_NT1NT2Phasecard_EstResp, self).__init__()
        self.encoder = EncoderNT1NT2Phasecard_EstiResp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)

class EncoderNT1NT2Coscard_EstiResp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderNT1NT2Coscard_EstiResp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        resp = mu[3]
        mu[0] = u1_normalized[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_cos[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = filter_resp(resp,50/60)
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var

class vae_NT1NT2Coscard_EstResp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_NT1NT2Coscard_EstResp, self).__init__()
        self.encoder = EncoderNT1NT2Coscard_EstiResp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)

class EncoderLT1NT2Coscard_EstiResp(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderLT1NT2Coscard_EstiResp, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, latent_dims)
        self.linear6 = nn.Linear(70, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        resp = mu[3]
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = u2_normalized[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        mu[2] = hidx_cos[t]
        log_var[2] = torch.log(torch.tensor(1e-6))
        mu[3] = filter_resp(resp,50/60)
        z = self.reparameterize(mu, log_var)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, log_var

class vae_LT1NT2Coscard_EstResp(nn.Module):
    def __init__(self, latent_dims):
        super(vae_LT1NT2Coscard_EstResp, self).__init__()
        self.encoder = EncoderLT1NT2Coscard_EstiResp(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)





class VariationalAutoencoder_hc(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder_hc, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)







# class VariationalAutoencoder_hc_m(nn.Module):
#     def __init__(self, latent_dims):
#         super(VariationalAutoencoder_hc_m, self).__init__()
#         self.encoder = EncoderModified(latent_dims)
#         self.decoder = Decoder(latent_dims)

#     def forward(self, x, t):
#         z, mu, log_var = self.encoder(x,t)
#         return self.decoder(z)
    
# class VariationalAutoencoder_linear4(nn.Module):
#     def __init__(self, latent_dims):
#         super(VariationalAutoencoder_linear4, self).__init__()
#         self.encoder = EncoderLinear4(latent_dims)
#         self.decoder = Decoder(latent_dims)

#     def forward(self, x, t):
#         z, mu, log_var = self.encoder(x,t)
#         return self.decoder(z)
    
# class VariationalAutoencoder_nonlinear4(nn.Module):
#     def __init__(self, latent_dims):
#         super(VariationalAutoencoder_nonlinear4, self).__init__()
#         self.encoder = EncoderNonlinear4(latent_dims)
#         self.decoder = Decoder(latent_dims)

#     def forward(self, x, t):
#         z, mu, log_var = self.encoder(x,t)
#         return self.decoder(z)
    



  
    

    

    


