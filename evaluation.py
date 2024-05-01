from matplotlib import pyplot as plt 
import numpy as np 
import torch 
from torch.utils.data import DataLoader
import os
import sys 

sys.path.insert(0, 'SADM-modified')
from DDPM import DDPM, ContextUnet
from ViVit import ViViT
from ACDC_loader import ACDCDataset

device = torch.device("cuda")
RESULT_DIR = "/media/7tb_encrypted/adriannas_project/results"
DATA_DIR = "/media/7tb_encrypted/adriannas_project"
MODEL_PATH = os.path.join(RESULT_DIR, 'model256_ep998.pth')


def compare_guide_w_samples(x_prev, model, guide_weights=[0.2, 1.0, 4.0]):
    num_plots = len(guide_weights)
    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots*6, 6))
    x_prev = x_prev.to(device)
    model = model.to(device)
    with torch.no_grad():
        for idx, guide_w in enumerate(guide_weights):
            x_gen, _ = model.sample(x_prev, device, guide_w)
            x_gen = x_gen.cpu()[0][0]
            x_gen = np.transpose(x_gen, (1, 2, 0))
            axes[idx].imshow(x_gen[:, :, x_gen.shape[2] // 2], cmap='gray')
            axes[idx].set_title(guide_w)
            axes[idx].axis('off')
    plt.savefig(f'{RESULT_DIR}/guide_w_comparison2.png', pad_inches=0)

def plot_slice_from_npy(npy_file):
    img = np.load(npy_file)[0][0]
    print(img.shape)
    img = np.transpose(img, (1, 2, 0))
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(img[:, :, img.shape[2] // 2], cmap='gray')
    file_name = npy_file.split('/')[-1].split('.')[0]
    plt.savefig(f'{RESULT_DIR}/{file_name}.jpg')

plot_slice_from_npy(f"{RESULT_DIR}/x_vivit2_499.npy")
plot_slice_from_npy(f"{RESULT_DIR}/x_vivit2_ep300.npy")
plot_slice_from_npy(f"{RESULT_DIR}/x_vivit2_ep400.npy")
plot_slice_from_npy(f"{RESULT_DIR}/x_vivit2_ep500.npy")

plot_slice_from_npy(f"{RESULT_DIR}/x_vivit2_ep100.npy")
plot_slice_from_npy(f"{RESULT_DIR}/x_vivit2_ep0.npy")

valid_loader = DataLoader(ACDCDataset(data_dir=DATA_DIR, split="tst"), batch_size=1, shuffle=False, num_workers=1)
image_size = (32, 128, 128)
num_frames = 3

n_T = 400  # 500
n_feat = 128 # 128 ok, 256 better (but slower)

patch_size = (8, 32, 32)
val_iter = iter(valid_loader)
x, x_prev = next(val_iter)
x_prev = x_prev.to(device)
for ep in [200, 300, 400, 499]:
    model_path = f'{RESULT_DIR}/vivit_ep{ep}.pth'
    vivit_model = ViViT(image_size, patch_size, num_frames)
    vivit_model.load_state_dict(torch.load(model_path))
    vivit_model.to(device)
    with torch.no_grad():
        out = vivit_model(x_prev)
    np.save(f'{RESULT_DIR}/vivit_gen_ep{ep}.npy', out.cpu())
    plot_slice_from_npy(f'{RESULT_DIR}/vivit_gen_ep{ep}.npy')


nn_model = ContextUnet(in_channels=1, n_feat=n_feat, in_shape=(1, *image_size))

ddpm = DDPM(vivit_model=vivit_model, nn_model=nn_model,
            betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)

ddpm.load_state_dict(torch.load(MODEL_PATH))
ddpm.to(device)
guide_w = 1 

val_iter = iter(valid_loader)
x, x_prev = next(val_iter)
with torch.no_grad(): 
    x_prev = x_prev.to(device)
    ddpm.vivit_model.to(device)
    print(x_prev.device)
    c = ddpm.vivit_model(x_prev)
    np.save(f"{RESULT_DIR}/context2.npy", c.cpu())
    plot_slice_from_npy(f"{RESULT_DIR}/context2.npy")

for i in range(5): 
    x, x_prev = next(val_iter)
    x_prev = x_prev.to(device)
    with torch.no_grad():
        x_gen, _ = ddpm.sample(x_prev, device, guide_w)
        np.save(f"{RESULT_DIR}/x_gen_1000epochs_{i}.npy", x_gen.cpu())
        plot_slice_from_npy(f"{RESULT_DIR}/x_gen_1000epochs_{i}.npy")

x, x_prev = next(val_iter)
x_prev = x_prev.to(device)
compare_guide_w_samples(x_prev, ddpm)