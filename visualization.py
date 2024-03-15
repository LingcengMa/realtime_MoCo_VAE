import os
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
# from models import VariationalAutoencoder_hc, VariationalAutoencoder_hc_m
import torchvision
import numpy as np
import scipy
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from PIL import Image
import imageio.v3 as iio

#Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#model
# vae4d_hc = VariationalAutoencoder_hc(latent_dims=4).to(device)
# vae4d_hc_f = VariationalAutoencoder_hc(latent_dims=4).to(device)
# vae4d_hc_m = VariationalAutoencoder_hc_m(latent_dims=4).to(device)
# vae4d_hc_m_f = VariationalAutoencoder_hc_m(latent_dims=4).to(device)

#data
data_file = os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'realtime_GT_imgs.mat')
mat = scipy.io.loadmat(data_file)
Phi_rt_init = mat['Phi_rt_init'] 
Phi_rt_init_real = Phi_rt_init.real
Phi_rt_init_imag = Phi_rt_init.imag
Phi_rt_init_con = np.concatenate((Phi_rt_init_real, Phi_rt_init_imag), axis = 1)
Phi_rt_init_con = Phi_rt_init_con.transpose()
# Phi_rt_init = torch.from_numpy(Phi_rt_init_con).float()
U_init = mat['U_init']


#data_feature
data_file= os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'feature.mat')
mat = scipy.io.loadmat(data_file)
Phi_rt_feature = mat['Phi_rt_feature'] 
Phi_rt_feature_real = Phi_rt_feature.real
Phi_rt_feature_imag = Phi_rt_feature.imag
Phi_rt_feature_con = np.concatenate((Phi_rt_feature_real, Phi_rt_feature_imag))
Phi_rt_feature_con = Phi_rt_feature_con.transpose()
# Phi_rt_feature = torch.from_numpy(Phi_rt_feature_con).float()

M_file= os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'M.mat')
mat = scipy.io.loadmat(M_file)
M = mat['M'] 

recon_LT1_LT2_Phase = np.load('recon_LT1_LT2_Phase.npy')
recon_NT1_NT2_Phase = np.load('recon_NT1_NT2_Phase.npy')

residual_base = recon_LT1_LT2_Phase - Phi_rt_feature_con
residual_best = recon_NT1_NT2_Phase - Phi_rt_feature_con

# #load output
# Phi_rt_init_hat = torch.load('Phi_rt_init_hat.pth')
# Phi_rt_init_hat = Phi_rt_init_hat.to('cpu').detach().numpy()

# Phi_rt_init_m_hat = torch.load('Phi_rt_init_m_hat.pth')
# Phi_rt_init_m_hat = Phi_rt_init_m_hat.to('cpu').detach().numpy()

# Phi_rt_feature_hat = torch.load('Phi_rt_feature_hat.pth')
# Phi_rt_feature_hat = Phi_rt_feature_hat.to('cpu').detach().numpy()

# Phi_rt_feature_m_hat = torch.load('Phi_rt_feature_m_hat.pth')
# Phi_rt_feature_m_hat = Phi_rt_feature_m_hat.to('cpu').detach().numpy()

# Phi_rt_feature_hat_fixed_linear = torch.load('Phi_rt_feature_hat_fixed_linear.pth')
# Phi_rt_feature_hat_fixed_linear = Phi_rt_feature_hat_fixed_linear.to('cpu').detach().numpy()

# Phi_rt_feature_hat_fixed_nonlinear = torch.load('Phi_rt_feature_hat_fixed_nonlinear.pth')
# Phi_rt_feature_hat_fixed_nonlinear = Phi_rt_feature_hat_fixed_nonlinear.to('cpu').detach().numpy()

# visualization
plt.figure(1)
plt.plot(Phi_rt_feature_con[0:1800,0], label="Input image signal")
plt.plot(recon_LT1_LT2_Phase[0:1800,0], label="Baseline Reconstruction")
plt.plot(recon_NT1_NT2_Phase[0:1800,0], label="Best Performing Reconstruction")
plt.xlabel("Time")
plt.ylabel("Phi_rt_feature[0]")
plt.legend()
plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/recon_compare_1.png')

plt.figure(2)
plt.plot(Phi_rt_feature_con[0:1800,1], label="Input Image Signal")
plt.plot(recon_LT1_LT2_Phase[0:1800,1], label="Baseline Reconstruction")
plt.plot(recon_NT1_NT2_Phase[0:1800,1], label="Best Performing Reconstruction")
plt.xlabel("Time")
plt.ylabel("Phi_rt_feature[1]")
plt.legend()
plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/recon_compare_2.png')

plt.figure(3)
plt.plot(Phi_rt_feature_con[0:1800,2], label="Input Image Signal")
plt.plot(recon_LT1_LT2_Phase[0:1800,2], label="Baseline Reconstruction")
plt.plot(recon_NT1_NT2_Phase[0:1800,2], label="Best Performing Reconstruction")
plt.xlabel("Time")
plt.ylabel("Phi_rt_feature[2]")
plt.legend()
plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/recon_compare_3.png')

plt.figure(4)
plt.plot(Phi_rt_feature_con[0:1800,3], label="Input Image Signal")
plt.plot(recon_LT1_LT2_Phase[0:1800,3], label="Baseline Reconstruction")
plt.plot(recon_NT1_NT2_Phase[0:1800,3], label="Best Performing Reconstruction")
plt.xlabel("Time")
plt.ylabel("Phi_rt_feature[3]")
plt.legend()
plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/recon_compare_4.png')

plt.figure(5)
plt.plot(residual_base[0:1800,0], label="Baseline Reconstruction")
plt.plot(residual_best[0:1800,0], label="Best Performing Reconstruction")
plt.axhline(y = 0, color = 'r', linestyle = 'dashed') 
plt.xlabel("Time")
plt.ylabel("Residuals[0]")
plt.legend()
plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/residual_compare_1.png')

plt.figure(6)
plt.plot(residual_base[0:1800,1], label="Baseline Reconstruction")
plt.plot(residual_best[0:1800,1], label="Best Performing Reconstruction")
plt.axhline(y = 0, color = 'r', linestyle = 'dashed') 
plt.xlabel("Time")
plt.ylabel("Residuals[1]")
plt.legend()
plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/residual_compare_2.png')

plt.figure(7)
plt.plot(residual_base[0:1800,2], label="Baseline Reconstruction")
plt.plot(residual_best[0:1800,2], label="Best Performing Reconstruction")
plt.axhline(y = 0, color = 'r', linestyle = 'dashed') 
plt.xlabel("Time")
plt.ylabel("Residuals[2]")
plt.legend()
plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/residual_compare_3.png')

plt.figure(8)
plt.plot(residual_base[0:1800,3], label="Baseline Reconstruction")
plt.plot(residual_best[0:1800,3], label="Best Performing Reconstruction")
plt.axhline(y = 0, color = 'r', linestyle = 'dashed') 
plt.xlabel("Time")
plt.ylabel("Residuals[3]")
plt.legend()
plt.savefig('/Users/xinguofang/Documents/Research/TI-aware Clustering/residual_compare_4.png')

#image
# recon_LT1_LT2_Phase_complex = recon_LT1_LT2_Phase[:, :48] + 1j * recon_LT1_LT2_Phase[:,48:96]
# recon_LT1_LT2_Phase_complex = recon_LT1_LT2_Phase_complex.transpose()
# # loss = Phi_rt_feature_hat_complex - Phi_rt_feature
# recon_NT1_NT2_Phase_complex = recon_NT1_NT2_Phase[:, :48] + 1j * recon_NT1_NT2_Phase[:,48:96]
# recon_NT1_NT2_Phase_complex = recon_NT1_NT2_Phase_complex.transpose()
# # loss_m = Phi_rt_feature_m_hat_complex - Phi_rt_feature

# # Phi_rt_feature_hat_fixed_linear_complex = Phi_rt_feature_hat_fixed_linear[:, :48] + 1j * Phi_rt_feature_hat_fixed_linear[:,48:96]
# # Phi_rt_feature_hat_fixed_linear_complex = Phi_rt_feature_hat_fixed_linear_complex.transpose()
# # Phi_rt_feature_hat_fixed_nonlinear_complex = Phi_rt_feature_hat_fixed_nonlinear[:, :48] + 1j * Phi_rt_feature_hat_fixed_nonlinear[:,48:96]
# # Phi_rt_feature_hat_fixed_nonlinear_complex = Phi_rt_feature_hat_fixed_nonlinear_complex.transpose()

# img_m = np.matmul(U_init, M)

# # img_original = np.abs(np.matmul(img_m, Phi_rt_feature[:,0:1800]))
# # img_recon_linear = np.abs(np.matmul(img_m, Phi_rt_feature_hat_complex[:,0:1800]))
# # img_recon_modified = np.abs(np.matmul(img_m, Phi_rt_feature_m_hat_complex[:,0:1800]))
# # img_loss_linear = np.abs(np.matmul(img_m, loss[:,0:1800]))
# # img_loss_modified = np.abs(np.matmul(img_m, loss_m[:,0:1800]))
# img_recon_base= np.abs(np.matmul(img_m, recon_LT1_LT2_Phase_complex[:,0:1800]))
# img_recon_best = np.abs(np.matmul(img_m, recon_NT1_NT2_Phase_complex[:,0:1800]))

# # img_recon_linear_fixed = np.abs(np.matmul(img_m, Phi_rt_feature_hat_fixed_linear_complex[:,0:1800]))
# # img_recon_nonlinear_fixed = np.abs(np.matmul(img_m, Phi_rt_feature_hat_fixed_nonlinear_complex[:,0:1800]))

# if not os.path.exists('output_images_recon_base'):
#     os.makedirs('output_images_recon_base')

# # Reshape the data into 1800 images (320x320 each)
# for i in range(1800):
#     image_data = img_recon_base[:, :, i]
#     plt.imshow(image_data)  # You can specify the desired colormap
#     plt.axis('off')  # Turn off axis labels
#     plt.savefig(f'output_images_recon_base/image_{i:04d}.png', bbox_inches='tight', pad_inches=0)
#     plt.close()

# # Create a list of image file paths
# image_files = ['output_images_recon_base/image_{:04d}.png'.format(i) for i in range(1800)]

# # Specify the output GIF file name
# output_gif = 'output_recon_base.gif'

# frames = np.stack([iio.imread(image_file) for image_file in image_files], axis=0)
# iio.imwrite(output_gif, frames, duration=5)


# if not os.path.exists('output_images_recon_best'):
#     os.makedirs('output_images_recon_best')

# Reshape the data into 1800 images (320x320 each)
# for i in range(1800):
#     image_data = img_recon_best[:, :, i]
#     plt.imshow(image_data)  # You can specify the desired colormap
#     plt.axis('off')  # Turn off axis labels
#     plt.savefig(f'output_images_recon_best/image_{i:04d}.png', bbox_inches='tight', pad_inches=0)
#     plt.close()

# # Create a list of image file paths
# image_files = ['output_images_recon_best/image_{:04d}.png'.format(i) for i in range(1800)]

# # Specify the output GIF file name
# output_gif = 'output_recon_best.gif'

# # Save the images as frames of a GIF
# frames = np.stack([iio.imread(image_file) for image_file in image_files], axis=0)
# iio.imwrite(output_gif, frames, duration=5)

# if not os.path.exists('output_images_recon_linear'):
#     os.makedirs('output_images_recon_linear')

# # Reshape the data into 1800 images (320x320 each)
# for i in range(1800):
#     image_data = img_recon_linear[:, :, i]
#     plt.imshow(image_data)  # You can specify the desired colormap
#     plt.axis('off')  # Turn off axis labels
#     plt.savefig(f'output_images_recon_linear/image_{i:04d}.png', bbox_inches='tight', pad_inches=0)
#     plt.close()

# # Create a list of image file paths
# image_files = ['output_images_recon_linear/image_{:04d}.png'.format(i) for i in range(1800)]

# # Specify the output GIF file name
# output_gif = 'output_recon_linear.gif'

# # Save the images as frames of a GIF
# with iio.get_writer(output_gif, mode='I', duration=10) as writer:
#     for image_file in image_files:
#         image = iio.imread(image_file)
#         writer.append_data(image)

# #-----------------------------------#
# if not os.path.exists('output_images_recon_modified'):
#     os.makedirs('output_images_recon_modified')

# # Reshape the data into 1800 images (320x320 each)
# for i in range(1800):
#     image_data = img_recon_modified[:, :, i]
#     plt.imshow(image_data)  # You can specify the desired colormap
#     plt.axis('off')  # Turn off axis labels
#     plt.savefig(f'output_images_recon_modified/image_{i:04d}.png', bbox_inches='tight', pad_inches=0)
#     plt.close()

# # Create a list of image file paths
# image_files = ['output_images_recon_modified/image_{:04d}.png'.format(i) for i in range(1800)]

# # Specify the output GIF file name
# output_gif = 'output_recon_modified.gif'

# # Save the images as frames of a GIF
# with iio.get_writer(output_gif, mode='I', duration=10) as writer:
#     for image_file in image_files:
#         image = iio.imread(image_file)
#         writer.append_data(image)

# image_original_list = [Image.fromarray(img_original[:, :, i]) for i in range(1800)]
# image_recon_linear_list = [Image.fromarray(img_recon_linear[:, :, i]) for i in range(1800)]
# image_recon_modified_list = [Image.fromarray(img_recon_modified[:, :, i]) for i in range(1800)]

# output_gif_original = 'original.gif'
# output_gif_linear = 'linear.gif'
# output_gif_modified = 'modified.gif'

# imageio.mimsave(output_gif_original, image_original_list, duration=10)
# imageio.mimsave(output_gif_linear, image_recon_linear_list, duration=10)
# imageio.mimsave(output_gif_modified, image_recon_modified_list, duration=10)

# #-----------------------------------------#
# if not os.path.exists('output_images_recon_linear_fixed'):
#     os.makedirs('output_images_recon_linear_fixed')

# # Reshape the data into 1800 images (320x320 each)
# for i in range(1800):
#     image_data = img_recon_linear_fixed[:, :, i]
#     plt.imshow(image_data)  # You can specify the desired colormap
#     plt.axis('off')  # Turn off axis labels
#     plt.savefig(f'output_images_recon_linear_fixed/image_{i:04d}.png', bbox_inches='tight', pad_inches=0)
#     plt.close()

# # Create a list of image file paths
# image_files = ['output_images_recon_linear_fixed/image_{:04d}.png'.format(i) for i in range(1800)]

# # Specify the output GIF file name
# output_gif = 'output_recon_linear_fixed.gif'

# # Save the images as frames of a GIF
# frames = np.stack([iio.imread(image_file) for image_file in image_files], axis=0)
# iio.imwrite(output_gif, frames)
# # with iio.get_writer(output_gif, mode='I', duration=10) as writer:
# #     for image_file in image_files:
# #         image = iio.imread(image_file)
# #         writer.append_data(image)

#-----------------------------------------#
# if not os.path.exists('output_images_recon_nonlinear_fixed'):
#     os.makedirs('output_images_recon_nonlinear_fixed')

# # Reshape the data into 1800 images (320x320 each)
# for i in range(1800):
#     image_data = img_recon_nonlinear_fixed[:, :, i]
#     plt.imshow(image_data)  # You can specify the desired colormap
#     plt.axis('off')  # Turn off axis labels
#     plt.savefig(f'output_images_recon_nonlinear_fixed/image_{i:04d}.png', bbox_inches='tight', pad_inches=0)
#     plt.close()

# # Create a list of image file paths
# image_files = ['output_images_recon_nonlinear_fixed/image_{:04d}.png'.format(i) for i in range(1800)]

# # Specify the output GIF file name
# output_gif = 'output_recon_nonlinear_fixed.gif'

# # Save the images as frames of a GIF
# frames = np.stack([iio.imread(image_file) for image_file in image_files], axis=0)
# iio.imwrite(output_gif, frames)
# # with iio.get_writer(output_gif, mode='I', duration=10) as writer:
# #     for image_file in image_files:
# #         image = iio.imread(image_file)
# #         writer.append_data(image)

# #-----------------------------------------#
# #subtraction
# if not os.path.exists('output_images_loss'):
#     os.makedirs('output_images_loss')

# # Reshape the data into 1800 images (320x320 each)
# for i in range(1800):
#     image_data = img_loss_linear[:, :, i]
#     plt.imshow(image_data)  # You can specify the desired colormap
#     plt.axis('off')  # Turn off axis labels
#     plt.savefig(f'output_images_loss/image_{i:04d}.png', bbox_inches='tight', pad_inches=0)
#     plt.close()

# # Create a list of image file paths
# image_files = ['output_images_loss/image_{:04d}.png'.format(i) for i in range(1800)]

# # Specify the output GIF file name
# output_gif = 'output_loss.gif'

# # Save the images as frames of a GIF
# frames = np.stack([iio.imread(image_file) for image_file in image_files], axis=0)
# iio.imwrite(output_gif, frames, duration=10)

# #-----------------------------------------#
# if not os.path.exists('output_images_loss_m'):
#     os.makedirs('output_images_loss_m')

# # Reshape the data into 1800 images (320x320 each)
# for i in range(1800):
#     image_data = img_loss_modified[:, :, i]
#     plt.imshow(image_data)  # You can specify the desired colormap
#     plt.axis('off')  # Turn off axis labels
#     plt.savefig(f'output_images_loss_m/image_{i:04d}.png', bbox_inches='tight', pad_inches=0)
#     plt.close()

# # Create a list of image file paths
# image_files = ['output_images_loss_m/image_{:04d}.png'.format(i) for i in range(1800)]

# # Specify the output GIF file name
# output_gif = 'output_loss_m.gif'

# # Save the images as frames of a GIF
# frames = np.stack([iio.imread(image_file) for image_file in image_files], axis=0)
# iio.imwrite(output_gif, frames, duration=10)




# image_files = ['output_images_original/image_{:04d}.png'.format(i) for i in range(1800)]

# output_gif = 'output_original.gif'

# frames = np.stack([iio.imread(image_file) for image_file in image_files], axis=0)
# iio.imwrite(output_gif, frames, duration=10)

# image_files = ['output_images_recon_linear/image_{:04d}.png'.format(i) for i in range(1800)]

# output_gif = 'output_recon_linear.gif'

# frames = np.stack([iio.imread(image_file) for image_file in image_files], axis=0)
# iio.imwrite(output_gif, frames, duration=10)

# image_files = ['output_images_recon_modified/image_{:04d}.png'.format(i) for i in range(1800)]

# output_gif = 'output_recon_modified.gif'

# frames = np.stack([iio.imread(image_file) for image_file in image_files], axis=0)
# iio.imwrite(output_gif, frames, duration=10)

# image_files = ['output_images_recon_nonlinear_fixed/image_{:04d}.png'.format(i) for i in range(1800)]

# output_gif = 'output_recon_nonlinear_fixed.gif'

# frames = np.stack([iio.imread(image_file) for image_file in image_files], axis=0)
# iio.imwrite(output_gif, frames, duration=10)

# image_files = ['output_images_recon_linear_fixed/image_{:04d}.png'.format(i) for i in range(1800)]

# output_gif = 'output_recon_linear_fixed.gif'

# frames = np.stack([iio.imread(image_file) for image_file in image_files], axis=0)
# iio.imwrite(output_gif, frames, duration=10)




