
# Generative Multitasking Implementation Instructions

## Installation

1. **Clone the python repository**
   ```bash
   git clone -b MATLABintegration --single-branch https://redlradadm05403.ad.medctr.ucla.edu:3000/xfang/vae_temporal_analysis.git
   ```

2. ## Clone the matlab repository**
   ```bash
   git clone https://redlradadm05403.ad.medctr.ucla.edu:3000/xfang/generativeMT.git
   reconstruction code is in `multitasking_recon_Generative.m`
   ```

3. **Set up a virtual environment** (optional/i didn't use a virtual env but it might be useful)

4. **Install dependencies**
   ```bash
   pip3 install -r requirements.txt
   ```
---

## Usage

### Training

```bash
python train_network.py <input_mat_path> <output_model_path>
```

### Inference

```bash
python inference.py --mat_file <mat_path> --model_path <model_path> --inference_path <inference_path>--output_dir <output_dir>
```

### Arguments

| Argument              | Description                                   |
|-----------------------|-----------------------------------------------|
| `--input_mat_path`    | Path to the saved `extracttopython.mat`       |
| `--output_model_path` | Path to the training output                   |
| `--mat_path`          | should be the same as input_mat_path          |
| `--model_path`        | Path to the saved `.pth` file                 |
| `--inference_path`    | Path to the `inference.mat`                   |
| `--output_dir`        | Directory where outputs will be saved         |

---

## Input Format


- `ExtractToPython.mat`: include all the parameters needed for training
- `inference.mat`: include the desired motion vectors and conditional inputs for inference


---

## Output

by train_network.py
- `trainingoutput.mat`: 
   1. latent space vectors: `latent_space`
   2. model path string: `model_path`
   3. realtime temporal basis phi (with eddy current): `recon_eddy_current`
   4. realtime temporal basis phi (eddy current corrected): `recon_wo_eddy_current`

by inference_network.py
- `reconoutput.mat`: 
   1. multidimentional temporal basis phi: `Phi_gen`


## Parameter tuning

- **HIGH NRMSE**: change KLD index, initialization manual seed, learning rate. 
