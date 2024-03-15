import os
import sys
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import scipy
from sklearn.neighbors import KernelDensity
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import entropy
from scipy.stats import gaussian_kde

#Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

binning_file = os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'binning_GT.mat')
binning_file = scipy.io.loadmat(binning_file)
hidx = binning_file['Hidx'] # Cardiac
ridx = binning_file['Ridx'] # Respiratory
ridx_mu = (ridx - ridx.mean())/ridx.std()
hidx_mu = (hidx - hidx.mean())/hidx.std()

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

# def calc_MI(x, y, num_bins):
#     # Discretize the data into bins
#     hist_x, bin_edges_x = np.histogram(x, bins=num_bins)
#     hist_y, bin_edges_y = np.histogram(y, bins=num_bins)
#     joint_probs = np.outer(hist_x, hist_y) / len(x)
#     marginal_x = hist_x / len(x)
#     marginal_y = hist_y / len(y)
#     entropy_x = entropy(marginal_x, base=2)
#     entropy_y = entropy(marginal_y, base=2)
#     joint_entropy = entropy(joint_probs.ravel(), base=2)
#     mutual_info = entropy_x + entropy_y - joint_entropy 
#     return mutual_info

# def calc_MI(x, y):
#     kd_x = gaussian_kde(x)
#     kd_y = gaussian_kde(y)
#     kd_xy = gaussian_kde(np.stack((x,y)))

#     #entropy calculations
#     h_xy = -np.mean(np.log2(kd_xy(np.stack((x,y)))))
#     h_x = -np.mean(np.log2(kd_x(x)))
#     h_y = -np.mean(np.log2(kd_y(y)))
    
#     mi = (h_x + h_y - h_xy)

#     return mi

def calc_MI(x,y):
    mi = mutual_info_regression(x,y)
    return mi




mutual_info_list1 = []

num_bins = 6

MI_Max = mutual_info_regression(ridx_mu.reshape(-1,1), ridx_mu.ravel())
print(MI_Max.item())

# latent_LT1_LT2_Sin_esti_resp = np.load('latent_LT1_LT2_Sin_esti_resp.npy')
# resp_LT1_LT2_Sin_esti = latent_LT1_LT2_Sin_esti_resp[:,3]
# # MI_LT1_LT2_Sin_esti = calc_MI(resp_LT1_LT2_Sin_esti.ravel(), ridx_mu.ravel(),num_bins)
# MI_LT1_LT2_Sin_esti = mutual_info_regression(ridx_mu.reshape(-1,1), resp_LT1_LT2_Sin_esti.ravel())
# MI_LT1_LT2_Sin_esti_norm = MI_LT1_LT2_Sin_esti/MI_Max
# # MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
# mutual_info_list2.append({"Model Name": 'LT1_LT2_Sin_esti_resp', "MI": MI_LT1_LT2_Sin_esti_norm})

# latent_LT1_LT2_Cos_esti_resp = np.load('latent_LT1_LT2_Cos_esti_resp.npy')
# resp_LT1_LT2_Cos_esti = latent_LT1_LT2_Cos_esti_resp[:,3]
# # MI_LT1_LT2_Sin_esti = calc_MI(resp_LT1_LT2_Sin_esti.ravel(), ridx_mu.ravel(),num_bins)
# MI_LT1_LT2_Cos_esti = mutual_info_regression(ridx_mu.reshape(-1,1), resp_LT1_LT2_Cos_esti.ravel())
# MI_LT1_LT2_Cos_esti_norm = MI_LT1_LT2_Cos_esti/MI_Max
# # MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
# mutual_info_list2.append({"Model Name": 'LT1_LT2_Cos_esti_resp', "MI": MI_LT1_LT2_Cos_esti_norm})

# latent_NT1_NT2_Phase_esti_resp = np.load('latent_NT1_NT2_Phase_esti_resp.npy')
# resp_NT1_NT2_Phase_esti = latent_NT1_NT2_Phase_esti_resp[:,3]
# # MI_LT1_LT2_Sin_esti = calc_MI(resp_LT1_LT2_Sin_esti.ravel(), ridx_mu.ravel(),num_bins)
# MI_NT1_NT2_Phase_esti = mutual_info_regression(ridx_mu.reshape(-1,1), resp_NT1_NT2_Phase_esti.ravel())
# MI_NT1_NT2_Phase_esti_norm = MI_NT1_NT2_Phase_esti/MI_Max
# # MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
# mutual_info_list2.append({"Model Name": 'NT1_NT2_Phase_esti_resp', "MI": MI_NT1_NT2_Phase_esti_norm})

# latent_NT1_NT2_Cos_esti_resp = np.load('latent_NT1_NT2_Cos_esti_resp.npy')
# resp_NT1_NT2_Cos_esti = latent_NT1_NT2_Cos_esti_resp[:,3]
# # MI_LT1_LT2_Sin_esti = calc_MI(resp_LT1_LT2_Sin_esti.ravel(), ridx_mu.ravel(),num_bins)
# MI_NT1_NT2_Cos_esti = mutual_info_regression(ridx_mu.reshape(-1,1), resp_NT1_NT2_Cos_esti.ravel())
# MI_NT1_NT2_Cos_esti_norm = MI_NT1_NT2_Cos_esti/MI_Max
# # MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
# mutual_info_list2.append({"Model Name": 'NT1_NT2_Cos_esti_resp', "MI": MI_NT1_NT2_Cos_esti_norm})

# latent_LT1_NT2_Cos_esti_resp = np.load('latent_LT1_NT2_Cos_esti_resp.npy')
# resp_LT1_NT2_Cos_esti = latent_LT1_NT2_Cos_esti_resp[:,3]
# # MI_LT1_LT2_Sin_esti = calc_MI(resp_LT1_LT2_Sin_esti.ravel(), ridx_mu.ravel(),num_bins)
# MI_LT1_NT2_Cos_esti = mutual_info_regression(ridx_mu.reshape(-1,1), resp_LT1_NT2_Cos_esti.ravel())
# MI_LT1_NT2_Cos_esti_norm = MI_LT1_NT2_Cos_esti/MI_Max
# # MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
# mutual_info_list2.append({"Model Name": 'LT1_NT2_Cos_esti_resp', "MI": MI_LT1_NT2_Cos_esti_norm})

# print(*mutual_info_list2)





# latent_LT1_LT2_Phase = np.load('latent_LT1_LT2_Phase.npy')
# resp_LT1_LT2_phase = latent_LT1_LT2_Phase[:,3]
# MI_LT1_LT2_Phase = calc_MI(ridx_mu.reshape(-1,1), resp_LT1_LT2_phase.ravel())
# # MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
# mutual_info_list1.append({"Model Name": 'LT1_LT2_Phase', "MI": MI_LT1_LT2_Phase/MI_Max})

# latent_NT1_LT2_Phase = np.load('latent_NT1_LT2_Phase.npy')
# resp_NT1_LT2_phase = latent_NT1_LT2_Phase[:,3]
# MI_NT1_LT2_Phase = calc_MI(ridx_mu.reshape(-1,1), resp_NT1_LT2_phase.ravel())
# # MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
# mutual_info_list1.append({"Model Name": 'NT1_LT2_Phase', "MI": MI_NT1_LT2_Phase/MI_Max})

# latent_NT1_NT2_Phase = np.load('latent_NT1_NT2_Phase.npy')
# resp_NT1_NT2_phase = latent_NT1_NT2_Phase[:,3]
# MI_NT1_NT2_Phase = calc_MI(ridx_mu.reshape(-1,1), resp_NT1_NT2_phase.ravel())
# # MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
# mutual_info_list1.append({"Model Name": 'NT1_NT2_Phase', "MI": MI_NT1_NT2_Phase/MI_Max})

# latent_LT1_NT2_Phase = np.load('latent_LT1_NT2_Phase.npy')
# resp_LT1_NT2_phase = latent_LT1_NT2_Phase[:,3]
# MI_LT1_NT2_Phase = calc_MI(ridx_mu.reshape(-1,1), resp_LT1_NT2_phase.ravel())
# # MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
# mutual_info_list1.append({"Model Name": 'LT1_NT2_Phase', "MI": MI_LT1_NT2_Phase/MI_Max})




latent_LT1_LT2_Sin = np.load('latent_LT1_LT2_Sin.npy')
resp_LT1_LT2_Sin = latent_LT1_LT2_Sin[:,3]
MI_LT1_LT2_Sin = mutual_info_regression(ridx_mu.reshape(-1,1), resp_LT1_LT2_Sin.ravel())
# MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
mutual_info_list1.append({"Model Name": 'LT1_LT2_Sin', "MI": MI_LT1_LT2_Sin/MI_Max})

latent_NT1_LT2_Sin = np.load('latent_NT1_LT2_Sin.npy')
resp_NT1_LT2_Sin = latent_NT1_LT2_Sin[:,3]
MI_NT1_LT2_Sin = mutual_info_regression(ridx_mu.reshape(-1,1), resp_NT1_LT2_Sin.ravel())
# MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
mutual_info_list1.append({"Model Name": 'NT1_LT2_Sin', "MI": MI_NT1_LT2_Sin/MI_Max})

latent_NT1_NT2_Sin = np.load('latent_NT1_NT2_Sin.npy')
resp_NT1_NT2_Sin = latent_NT1_NT2_Sin[:,3]
MI_NT1_NT2_Sin = mutual_info_regression(ridx_mu.reshape(-1,1), resp_NT1_NT2_Sin.ravel())
# MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
mutual_info_list1.append({"Model Name": 'NT1_NT2_Sin', "MI": MI_NT1_NT2_Sin/MI_Max})

latent_LT1_NT2_Sin = np.load('latent_LT1_NT2_Sin.npy')
resp_LT1_NT2_Sin = latent_LT1_NT2_Sin[:,3].reshape(-1, 1)
MI_LT1_NT2_Sin = mutual_info_regression(ridx_mu.reshape(-1,1), resp_LT1_NT2_Sin.ravel())
# MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
mutual_info_list1.append({"Model Name": 'LT1_NT2_Sin', "MI": MI_LT1_NT2_Sin/MI_Max})




# latent_LT1_LT2_Cos = np.load('latent_LT1_LT2_Cos.npy')
# resp_LT1_LT2_Cos = latent_LT1_LT2_Cos[:,3]
# MI_LT1_LT2_Cos = calc_MI(ridx_mu.reshape(-1,1), resp_LT1_LT2_Cos.ravel())
# # MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
# mutual_info_list1.append({"Model Name": 'LT1_LT2_Cos', "MI": MI_LT1_LT2_Cos/MI_Max})

# latent_NT1_LT2_Cos = np.load('latent_NT1_LT2_Cos.npy')
# resp_NT1_LT2_Cos = latent_NT1_LT2_Cos[:,3].reshape(-1, 1)
# MI_NT1_LT2_Cos = calc_MI(ridx_mu.reshape(-1,1), resp_NT1_LT2_Cos.ravel())
# # MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
# mutual_info_list1.append({"Model Name": 'NT1_LT2_Cos', "MI": MI_NT1_LT2_Cos/MI_Max})

# latent_NT1_NT2_Cos = np.load('latent_NT1_NT2_Cos.npy')
# resp_NT1_NT2_Cos = latent_NT1_NT2_Cos[:,3].reshape(-1, 1)
# MI_NT1_NT2_Cos = calc_MI(ridx_mu.reshape(-1,1), resp_NT1_NT2_Cos.ravel())
# # MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
# mutual_info_list1.append({"Model Name": 'NT1_NT2_Cos', "MI": MI_NT1_NT2_Cos/MI_Max})

# latent_LT1_NT2_Cos = np.load('latent_LT1_NT2_Cos.npy')
# resp_LT1_NT2_Cos = latent_LT1_NT2_Cos[:,3].reshape(-1, 1)
# MI_LT1_NT2_Cos = calc_MI(ridx_mu.reshape(-1,1), resp_LT1_NT2_Cos.ravel())
# # MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
# mutual_info_list1.append({"Model Name": 'LT1_NT2_Cos', "MI": MI_LT1_NT2_Cos/MI_Max})



# latent_LT1_LT2_CosSin = np.load('latent_LT1_LT2_CosSin.npy')
# resp_LT1_LT2_CosSin = latent_LT1_LT2_CosSin[:,4]
# MI_LT1_LT2_CosSin = calc_MI(ridx_mu.reshape(-1,1), resp_LT1_LT2_CosSin.ravel())
# # MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
# mutual_info_list1.append({"Model Name": 'LT1_LT2_CosSin', "MI": MI_LT1_LT2_CosSin/MI_Max})

# latent_NT1_LT2_CosSin = np.load('latent_NT1_LT2_CosSin.npy')
# resp_NT1_LT2_CosSin = latent_NT1_LT2_CosSin[:,4]
# MI_NT1_LT2_CosSin = calc_MI(ridx_mu.reshape(-1,1), resp_NT1_LT2_CosSin.ravel())
# # MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
# mutual_info_list1.append({"Model Name": 'NT1_LT2_CosSin', "MI": MI_NT1_LT2_CosSin/MI_Max})

# latent_NT1_NT2_CosSin = np.load('latent_NT1_NT2_CosSin.npy')
# resp_NT1_NT2_CosSin = latent_NT1_NT2_CosSin[:,4]
# MI_NT1_NT2_CosSin = calc_MI(ridx_mu.reshape(-1,1), resp_NT1_NT2_CosSin.ravel())
# # MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
# mutual_info_list1.append({"Model Name": 'NT1_NT2_CosSin', "MI": MI_NT1_NT2_CosSin/MI_Max})

# latent_LT1_NT2_CosSin = np.load('latent_LT1_NT2_CosSin.npy')
# resp_LT1_NT2_CosSin = latent_LT1_NT2_CosSin[:,4]
# MI_LT1_NT2_CosSin = calc_MI(ridx_mu.reshape(-1,1), resp_LT1_NT2_CosSin.ravel())
# # MI = MutualInformation(resp_LT1_LT2_phase, ridx_mu, num_bins)
# mutual_info_list1.append({"Model Name": 'LT1_NT2_CosSin', "MI": MI_LT1_NT2_CosSin/MI_Max})

print(*mutual_info_list1)



