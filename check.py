import torch
from network import CVAE_BP_HT
import math
import matplotlib.pyplot as plt

# Create a step signal
x = torch.zeros(100, device="cuda")
x[50:] = 1.0  # step at index 50

model = CVAE_BP_HT(feature_size=98, latent_dims=1, class_size=1)
# Assume model is on CUDA as well

# Test your band-pass filter
y = model.band_pass_filter(x, lower_limit=0.67, upper_limit=2.5)
# Print or plot first few and last few samples
print("Input (first 10):", x[:10])
print("Output (first 10):", y[:10])
print("Input (last 10):", x[-10:])
print("Output (last 10):", y[-10:])


fs = 10  # sampling rate in Hz
t = torch.linspace(0, 1, steps=fs, device="cuda")  # 1 second
x_sine = torch.sin(2 * math.pi * 1.0 * t)  # 1 Hz

y_sine = model.low_pass_filter(x_sine, upper_limit=0.67)
print("Original sine (first few):", x_sine[:5])
print("Filtered sine (first few):", y_sine[:5])

x_cpu = x_sine.cpu().numpy()
y_cpu = y_sine.cpu().numpy()

plt.figure()
plt.plot(x_cpu, label="Original")
plt.plot(y_cpu, label="Filtered")
plt.legend()
plt.title("Comparing Original and Filtered Signals")
plt.savefig("filter.png")
