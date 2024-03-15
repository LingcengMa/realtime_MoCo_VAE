import torch; 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        """you generate a random disteribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        var = self.linear6(x)
        sigma = torch.exp(var)
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return mu, var
          
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
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    



  
    

    

    


