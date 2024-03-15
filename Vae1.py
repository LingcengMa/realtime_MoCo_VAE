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
from torchmetrics.functional.clustering import mutual_info_score

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

class Encoder1(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder1, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 70)
        self.linear5 = nn.Linear(70, 10)
        self.linear6 = nn.Linear(10, latent_dims)
        self.linear7 = nn.Linear(10, latent_dims)

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
        x = F.relu(self.linear5(x))
        mu =  self.linear6(x)
        log_var = self.linear7(x)
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
    
class Decoder1(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder1, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dims,10),
            nn.ReLU(),
            nn.Linear(10,70),
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
    


class vae_test1(nn.Module):
    def __init__(self, latent_dims):
        super(vae_test1, self).__init__()
        self.encoder = Encoder1(latent_dims)
        self.decoder = Decoder1(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)


class Encoder2(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder2, self).__init__()
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
    
class Decoder2(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder2, self).__init__()
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
    


class vae_test2(nn.Module):
    def __init__(self, latent_dims):
        super(vae_test2, self).__init__()
        self.encoder = Encoder2(latent_dims)
        self.decoder = Decoder2(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)



class Encoder3(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder3, self).__init__()
        self.linear1 = nn.Linear(96, 192)
        self.linear2 = nn.Linear(192, 384)
        self.linear3 = nn.Linear(384, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, latent_dims)
        self.linear6 = nn.Linear(64, latent_dims)

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
    
class Decoder3(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder3, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dims,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 96)
        )

    def forward(self, z):
        return self.layers(z)
    


class vae_test3(nn.Module):
    def __init__(self, latent_dims):
        super(vae_test3, self).__init__()
        self.encoder = Encoder3(latent_dims)
        self.decoder = Decoder3(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)


class Encoder4(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder4, self).__init__()
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
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        x = F.leaky_relu(self.linear4(x))
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
    
class Decoder4(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder4, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dims,70),
            nn.LeakyReLU(),
            nn.Linear(70,100),
            nn.LeakyReLU(),
            nn.Linear(100,50),
            nn.LeakyReLU(),
            nn.Linear(50, 25),
            nn.LeakyReLU(),
            nn.Linear(25, 96)
        )

    def forward(self, z):
        return self.layers(z)
    


class vae_test4(nn.Module):
    def __init__(self, latent_dims):
        super(vae_test4, self).__init__()
        self.encoder = Encoder4(latent_dims)
        self.decoder = Decoder4(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)


class Encoder5(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder5, self).__init__()
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
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = F.tanh(self.linear3(x))
        x = F.tanh(self.linear4(x))
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
    
class Decoder5(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder5, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dims,70),
            nn.Tanh(),
            nn.Linear(70,100),
            nn.Tanh(),
            nn.Linear(100,50),
            nn.Tanh(),
            nn.Linear(50, 25),
            nn.Tanh(),
            nn.Linear(25, 96)
        )

    def forward(self, z):
        return self.layers(z)
    


class vae_test5(nn.Module):
    def __init__(self, latent_dims):
        super(vae_test5, self).__init__()
        self.encoder = Encoder5(latent_dims)
        self.decoder = Decoder5(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)


class Encoder6(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder6, self).__init__()
        self.linear1 = nn.Linear(96, 25)
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 100)
        self.linear4 = nn.Linear(100, 200)
        self.linear5 = nn.Linear(200, 400)
        self.linear6 = nn.Linear(400, 300)
        self.linear7 = nn.Linear(300, 150)
        self.linear8 = nn.Linear(150, 70)
        self.linear9 = nn.Linear(70, latent_dims)
        self.linear10 = nn.Linear(70, latent_dims)

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
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        x = F.leaky_relu(self.linear4(x))
        x = F.leaky_relu(self.linear5(x))
        x = F.leaky_relu(self.linear6(x))
        x = F.leaky_relu(self.linear7(x))
        x = F.leaky_relu(self.linear8(x))
        mu =  self.linear9(x)
        log_var = self.linear10(x)
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
    
class Decoder6(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder6, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dims,70),
            nn.LeakyReLU(),
            nn.Linear(70,150),
            nn.LeakyReLU(),
            nn.Linear(150,300),
            nn.LeakyReLU(),
            nn.Linear(300, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 25),
            nn.LeakyReLU(),
            nn.Linear(25, 96)
        )

    def forward(self, z):
        return self.layers(z)
    


class vae_test6(nn.Module):
    def __init__(self, latent_dims):
        super(vae_test6, self).__init__()
        self.encoder = Encoder6(latent_dims)
        self.decoder = Decoder6(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)





class Encoder7(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder7, self).__init__()
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
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        x = F.leaky_relu(self.linear4(x))
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
    
class Decoder7(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder7, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dims,70),
            nn.LeakyReLU(),
            nn.Linear(70,100),
            nn.LeakyReLU(),
            nn.Linear(100,50),
            nn.LeakyReLU(),
            nn.Linear(50, 25),
            nn.LeakyReLU(),
            nn.Linear(25, 96)
        )

    def forward(self, z):
        return self.layers(z)

class vae_test7(nn.Module):
    def __init__(self, latent_dims):
        super(vae_test7, self).__init__()
        self.encoder = Encoder7(latent_dims)
        self.decoder = Decoder7(latent_dims)

        self.apply(self._init_weights)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    



# class VGG(nn.Module):
#     def __init__(self, channel_in=3, width=64):
#         super(VGG, self).__init__()

#         self.conv1 = nn.Conv2d(channel_in, width, 3, 1, 1)
#         self.conv2 = nn.Conv2d(width, width, 3, 1, 1)

#         self.conv3 = nn.Conv2d(width, 2 * width, 3, 1, 1)
#         self.conv4 = nn.Conv2d(2 * width, 2 * width, 3, 1, 1)

#         self.conv5 = nn.Conv2d(2 * width, 4 * width, 3, 1, 1)
#         self.conv6 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
#         self.conv7 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
#         self.conv8 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)

#         self.conv9 = nn.Conv2d(4 * width, 8 * width, 3, 1, 1)
#         self.conv10 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
#         self.conv11 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
#         self.conv12 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)

#         self.conv13 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
#         self.conv14 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
#         self.conv15 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
#         self.conv16 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)

#         self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.relu = nn.ReLU()

#         self.load_params_()


class Encoder8(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder8, self).__init__()
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
        self.mi = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, t):
        x = x.to(device)
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        x = F.leaky_relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        sigma = torch.exp(log_var)
        mu[0] = segidx_mu[t]
        log_var[0] = torch.log(torch.tensor(1e-6))
        mu[1] = wall_clock_mu[t]
        log_var[1] = torch.log(torch.tensor(1e-6))
        z = self.reparameterize(mu, log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        # self.mi = mutual_info_score(mu[0],mu[2]) + mutual_info_score(mu[0],mu[3]) + mutual_info_score(mu[1],mu[2]) + mutual_info_score(mu[1],mu[3]) + mutual_info_score(mu[2],mu[3])
        return z, mu, log_var
    
    # the size of mu and kl

class Decoder8(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder8, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dims,70),
            nn.LeakyReLU(),
            nn.Linear(70,100),
            nn.LeakyReLU(),
            nn.Linear(100,50),
            nn.LeakyReLU(),
            nn.Linear(50, 25),
            nn.LeakyReLU(),
            nn.Linear(25, 96)
        )

    def forward(self, z):
        return self.layers(z)

class vae_test8(nn.Module):
    def __init__(self, latent_dims):
        super(vae_test8, self).__init__()
        self.encoder = Encoder8(latent_dims)
        self.decoder = Decoder8(latent_dims)

    def forward(self, x, t):
        z, mu, log_var = self.encoder(x,t)
        return self.decoder(z)