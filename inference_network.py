import sys
import torch

# torch.manual_seed(0)
import torch.utils
import torch.distributions
from network import CVAE_BP_HT
import scipy
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200
torch.autograd.set_detect_anomaly(True)

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# getting input and output paths from matlab
input_mat = sys.argv[1]
model_path = sys.argv[2]
inference_path = sys.argv[3]
recon_output_path = sys.argv[4]


def inference(mat_path, model_path, inference_path):
    # Load the .mat file (saved with -v7.3 in MATLAB)
    data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    inference = scipy.io.loadmat(
        inference_path, struct_as_record=False, squeeze_me=True
    )

    ExtractToPython = data["ExtractToPython"]
    # Extract variables (examples below, adapt as needed)
    Phi_rt_init = (
        torch.from_numpy(ExtractToPython.Phi_rt_init.transpose()).float().to(device)
    )
    motion_vector = torch.from_numpy(inference["motion_vector"]).float().to(device)
    md_inference_condition = (
        torch.from_numpy(inference["md_inference_condition"]).float().to(device)
    )
    feature_size = Phi_rt_init.size()[1]
    latent_dim = ExtractToPython.NumofLatent
    class_size = ExtractToPython.NumofCon
    sampling_rate = ExtractToPython.sampling_rate
    LPF = ExtractToPython.LPF
    BPFlb = ExtractToPython.BPFlb
    BPFub = ExtractToPython.BPFub

    model = CVAE_BP_HT(
        feature_size, latent_dim, class_size, sampling_rate, LPF, BPFlb, BPFub
    ).to(device)

    # Load the trained model parameters

    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        recon_x = model.decode(motion_vector, md_inference_condition)

    Phi_gen = recon_x.to("cpu").detach().numpy()

    return Phi_gen


Phi_gen = inference(input_mat, model_path, inference_path)

scipy.io.savemat(
    recon_output_path,
    {
        "Phi_gen": Phi_gen,
    },
)
