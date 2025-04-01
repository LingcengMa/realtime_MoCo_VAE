import torch
import numpy as np
from module import CVAE

def model_load(path):
    model = CVAE(108, latent_dims=2, class_size=4)
    model.load_state_dict(torch.load(path), map_location=torch.device('cpu'))
    model.eval()
    return model