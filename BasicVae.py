import os
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import scipy
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.ReLU(),
            nn.Linear(hidden_dims[3], z_dim * 2)  
        )

    def forward(self, x):
        encoded = self.encoder(x)
        mean, log_var = encoded[:, :z_dim], encoded[:, z_dim:]
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dims, output_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dims[3]),
            nn.ReLU(),
            nn.Linear(hidden_dims[3], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.decoder(x)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims, z_dim)
        self.decoder = Decoder(z_dim, hidden_dims, input_dim)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        reconstructed = self.decoder(z)
        return reconstructed, mean, log_var
    
# Define hyperparameters
input_dim = 96 
hidden_dims = [25, 50, 100, 70]  # Hidden dimensions of MLP layers
lr = 0.001
z_dim = 1
batch_size = 128
epochs = 50

#data
data_file = os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'realtime_GT_imgs.mat')
mat = scipy.io.loadmat(data_file)
data = mat['Phi_rt_init'] 
data_real = data.real
data_imag = data.imag
data_con = np.concatenate((data_real, data_imag))
data_con = data_con.transpose()
data = torch.from_numpy(data_con).float()
train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)


# print(data_con.shape)

seg_file = os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'TI.mat')
segidx = scipy.io.loadmat(seg_file)
segidx = segidx['Segidx']

binning_file = os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'binning_GT.mat')
binning_file = scipy.io.loadmat(binning_file)
hidx = binning_file['Hidx'] # Cardiac
ridx = binning_file['Ridx'] # Respiratory

# loss function
def vae_loss(reconstructed, x, mean, log_var):
    reconstruction_loss = nn.MSELoss(reconstructed, x)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence

# Instantiate VAE model and optimizer
model_1d = VAE(input_dim, hidden_dims, z_dim)
optimizer = optim.Adam(model_1d.parameters(), lr=lr)

if __name__ == '__main__':
    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, input_dim)  # Flatten input images
            optimizer.zero_grad()
            reconstructed, mean, log_var = model_1d(data)
            loss = vae_loss(reconstructed, data, mean, log_var)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
    z1 = mean.detach().numpy()
    np.save('latent_vectors_z1.npy', z1)
    torch.save(model_1d.state_dict(), 'vae1d_weights.pth')





