import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)

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
        #
        # self.mi = 0

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mu =  self.linear5(x)
        log_var = self.linear6(x)
        sigma = torch.exp(log_var)
        z = self.reparameterize(mu, log_var)
        self.kl = - 0.5 * torch.sum(1+ log_var - mu**2 - log_var.exp())
        #
        # self.mi = mi_score(mu[0],mu[2]) + mi_score(mu[0],mu[3]) + mi_score(mu[1],mu[2]) + mi_score(mu[1],mu[3]) + mi_score(mu[2],mu[3])
        return z
    
    # the size of mu and kl
    
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
    
class VAE(nn.Module):
    def __init__(self, latent_dims):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
class ConditionalVAE(nn.Module):
    def __init__(self, latent_dims, known_dims):
        super().__init__()
        # Add a linear layer for the class label
        self.label_projector = nn.Sequential(
            nn.Linear(known_dims, latent_dims),
            nn.ReLU(),
        )
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)
    
    def condition_vectors(self, z, y):
        projected_label = self.label_projector(y.float())
        return z + projected_label
    
    def forward(self, x, y):
        # Pass the input through the encoder
        encoded = self.encoder(x)
        # Pass the latent variable through the decoder
        decoded = self.decoder(self.condition_on_label(encoded, y))
        # Return the encoded output, decoded output, mean, and log variance
        return encoded, decoded


class CVAE(nn.Module):
    def __init__(self, feature_size, latent_dims, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # encode
        self.fc1 = nn.Linear(feature_size + class_size, 25)
        self.fc2 = nn.Linear(25, 50)
        self.fc3 = nn.Linear(50, 100)
        self.fc4 = nn.Linear(100, 70)
        self.fc51 = nn.Linear(70, latent_dims)
        self.fc52 = nn.Linear(70, latent_dims)

        # decode
        self.fc6 = nn.Linear(latent_dims + class_size, 70)
        self.fc7 = nn.Linear(70, 100)
        self.fc8 = nn.Linear(100, 50)
        self.fc9 = nn.Linear(50, 25)
        self.fc10 = nn.Linear(25, feature_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        h1 = self.relu(self.fc1(inputs))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        h4 = self.relu(self.fc4(h3))
        z_mu = self.fc51(h4)
        z_var = self.fc52(h4)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        h6 = self.relu(self.fc6(inputs))
        h7 = self.relu(self.fc7(h6))
        h8 = self.relu(self.fc8(h7))
        h9 = self.relu(self.fc9(h8))
        return self.sigmoid(self.fc10(h9))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
    
# input zero to the network as test
class CVAEtest(nn.Module):
    def __init__(self, feature_size, latent_dims, class_size):
        super(CVAEtest, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size
        self.latent_dims = latent_dims

        # encode
        self.fc1 = nn.Linear(feature_size + class_size, 25)
        self.fc2 = nn.Linear(25, 50)
        self.fc3 = nn.Linear(50, 100)
        self.fc4 = nn.Linear(100, 70)
        self.fc51 = nn.Linear(70, latent_dims)
        self.fc52 = nn.Linear(70, latent_dims)

        # decode
        self.fc6 = nn.Linear(latent_dims + class_size, 70)
        self.fc7 = nn.Linear(70, 100)
        self.fc8 = nn.Linear(100, 50)
        self.fc9 = nn.Linear(50, 25)
        self.fc10 = nn.Linear(25, feature_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        h1 = self.relu(self.fc1(inputs))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        h4 = self.relu(self.fc4(h3))
        z_mu = self.fc51(h4)
        z_var = self.fc52(h4)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        z_zeros = torch.zeros_like(c[:, :self.latent_dims])

        inputs = torch.cat([z_zeros, c], 1) # (bs, latent_size+class_size)
        h6 = self.relu(self.fc6(inputs))
        h7 = self.relu(self.fc7(h6))
        h8 = self.relu(self.fc8(h7))
        h9 = self.relu(self.fc9(h8))
        return self.sigmoid(self.fc10(h9))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(c), mu, logvar
