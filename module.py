import torch
from torch import nn


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
        self.fc6 = nn.Linear(latent_dims + class_size + 1, 70)
        self.fc7 = nn.Linear(70, 100)
        self.fc8 = nn.Linear(100, 50)
        self.fc9 = nn.Linear(50, 25)
        self.fc10 = nn.Linear(25, feature_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def band_pass_filter(self, z, lower_limit, upper_limit):
        d = 0.00712  # sampling rate
        num_extend = torch.ceil(torch.tensor(1 / (2 * d * upper_limit))).int()
        extended_z = torch.cat(
            [
                torch.full((num_extend,), z[0].item(), device=z.device),
                z,
                torch.full((num_extend,), z[-1].item(), device=z.device),
            ],
            dim=0,
        )
        z_fft = torch.fft.fft(extended_z, dim=-1)
        z_freq = torch.fft.fftfreq(len(extended_z), d=d, device=z.device)
        z_f = torch.where(
            (abs(z_freq) >= lower_limit) & (abs(z_freq) <= upper_limit),
            z_fft,
            torch.zeros_like(z_fft),
        )
        z_filtered = torch.fft.ifft(z_f, dim=-1).real[num_extend:-num_extend]
        return z_filtered

    def low_pass_filter(self, z, upper_limit):  # extend the input
        # Keep only the latent variables within the specified frequency band
        # extend the input at the beginning and end
        # extend by X = ceil(1/(2*d*upper_limit))
        d = 0.00712  # sampling rate
        num_extend = torch.ceil(torch.tensor(1 / (2 * d * upper_limit))).int()
        extended_z = torch.cat(
            [
                torch.full((num_extend,), z[0].item(), device=z.device),
                z,
                torch.full((num_extend,), z[-1].item(), device=z.device),
            ],
            dim=0,
        )
        z_fft = torch.fft.fft(extended_z, dim=-1)
        z_freq = torch.fft.fftfreq(len(extended_z), d=d, device=z.device)
        z_f = torch.where((abs(z_freq) <= upper_limit), z_fft, torch.zeros_like(z_fft))
        z_filtered = torch.fft.ifft(z_f, dim=-1).real[num_extend:-num_extend]
        return z_filtered

    def hilbert_transform(self, signal):
        # Compute the FFT of the input signal
        N = signal.shape[0]
        signal_fft = torch.fft.fft(signal)
        # Construct the Hilbert transform filter in the frequency domain
        h = torch.zeros(N, dtype=torch.complex64, device="cuda")
        if N % 2 == 0:
            h[0] = 1
            h[1 : N // 2] = 2
            h[N // 2] = 1
        else:
            h[0] = 1
            h[1 : (N + 1) // 2] = 2
        # Apply the filter to the FFT of the signal
        hilbert_fft = signal_fft * h
        # Compute the inverse FFT to obtain the analytic signal
        analytic_signal = torch.fft.ifft(hilbert_fft)
        # The real part of the analytic signal is the original signal
        # The imaginary part is the Hilbert transform of the original signal
        hilbert_transformed_signal = analytic_signal.imag
        return hilbert_transformed_signal

    def encode(self, x, c):  # Q(z|x, c)
        """
        x: (bs, feature_size)
        c: (bs, class_size)
        """
        inputs = torch.cat([x, c], 1)  # (bs, feature_size+class_size)
        h1 = self.relu(self.fc1(inputs))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        h4 = self.relu(self.fc4(h3))
        z_mu = self.fc51(h4)
        z_var = self.fc52(h4)
        u1 = z_mu[:, 0]
        u2 = z_mu[:, 1]
        u1_filtered = self.low_pass_filter(u1, 0.67)
        u2_filtered = self.band_pass_filter(u2, 0.67, 2.5)
        u3 = self.hilbert_transform(u2_filtered)
        z_mu = torch.stack((u1_filtered, u2_filtered, u3), dim=1)
        z_var = torch.stack(
            (z_var[:, 0], z_var[:, 1], z_var[:, 1]), dim=1
        )  # set the variance of the Hilbert transform to var2
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):  # P(x|z, c)
        """
        z: (bs, latent_size)
        c: (bs, class_size)
        """
        inputs = torch.cat([z, c], 1)  # (bs, latent_size+class_size)
        h6 = self.relu(self.fc6(inputs))
        h7 = self.relu(self.fc7(h6))
        h8 = self.relu(self.fc8(h7))
        h9 = self.relu(self.fc9(h8))
        return self.fc10(h9)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar