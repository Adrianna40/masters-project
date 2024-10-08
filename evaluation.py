from matplotlib import pyplot as plt 
import numpy as np 
import nibabel as nib
import torch 
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import sys 
import scipy.ndimage
import wandb
import yaml
from tqdm import tqdm


sys.path.insert(0, 'SADM-modified')
from DDPM import DDPM, ContextUnet, SingleDDPM
from ViVit import ViViT
from ACDC_loader import ACDCDataset

device = torch.device("cuda")
RESULT_DIR = "/media/7tb_encrypted/adriannas_project/results"
DATA_DIR = "/media/7tb_encrypted/adriannas_project"
# MODEL_PATH = f'{RESULT_DIR}/ddpm_single_ep999.pth'
MODEL_PATH = f'{RESULT_DIR}/pretrained_combined_499.pth'
# MODEL_PATH = f'{RESULT_DIR}/model256_ep998.pth'
IMG_SHAPE = (384, 384, 64)
TARGET_SHAPE = (128, 128, 32)
reversed_zoom = tuple([IMG_SHAPE[i]/TARGET_SHAPE[i] for i in range(len(IMG_SHAPE))])


def compare_guide_w_samples(x_prev, model, file_name, guide_weights=[0.2, 1.0, 4.0]):
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
    plt.savefig(f'{RESULT_DIR}/{file_name}', pad_inches=0)

def plot_slice_from_npy(npy_file):
    img = np.load(npy_file)[0][0]
    print(img.shape)
    img = np.transpose(img, (1, 2, 0))
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(img[:, :, img.shape[2] // 2], cmap='gray')
    file_name = npy_file.split('/')[-1].split('.')[0]
    plt.savefig(f'{RESULT_DIR}/{file_name}.jpg')

def load_to_nifti(npy_file_path):
    arr = np.load(npy_file_path)[0][0]
    arr = np.transpose(arr, (2, 1, 0))
    arr = scipy.ndimage.zoom(arr, reversed_zoom, order=1)
    nifti_img = nib.Nifti1Image(arr, affine=np.eye(4))
    file_name = npy_file_path.split('/')[-1].split('.')[0]
    nib.save(nifti_img, f'{RESULT_DIR}/{file_name}.nii.gz')

def evaluate_sampling(ddpm_instance, model_path, val_loader, guide_w=1.0):
    ddpm_instance.load_state_dict(torch.load(model_path))
    ddpm_instance.to(device)
    ddpm_instance.eval()
    loss = nn.MSELoss()
    losses = []
    save_img = True 
    with torch.no_grad():
        vbar = tqdm(val_loader)
        for x, x_prev in vbar:
            x = x.to(device)
            x_prev = x_prev.to(device)
            x_gen, _ = ddpm_instance.sample(x_prev, device, guide_w)
            batch_loss = loss(x, x_gen).cpu()
            print(batch_loss)
            losses.append(batch_loss)
            if save_img:
                file_name = model_path.split('/')[-1]
                file_path = f'{RESULT_DIR}/ {file_name}_2.npy'
                np.save(file_path, x_gen.cpu())
                plot_slice_from_npy(file_path)
                load_to_nifti(file_path)
                save_img = False 
    return losses
            
def evaluate_across_epochs(ddpm_instance, model_path_no_ep, val_loader, epochs):
    with open('local_config.yml', 'r') as f:
        local_user_config = yaml.safe_load(f)
    project = local_user_config['project']
    entity = local_user_config['entity']
    wandb.init(project, entity)
    results = {}
    for ep in epochs:
        model_path = f'{model_path_no_ep}{ep}.pth'
        losses = evaluate_sampling(ddpm_instance, model_path, val_loader)
        wandb.log({'epoch': ep, 'val_sampling_loss_avg': np.mean(losses), 'val_sampling_loss_std': np.std(losses), 'val_sampling_loss_list': losses})
        results[ep] = losses
    return results

# plot_slice_from_npy(f"{RESULT_DIR}/x_ddpm_single_999.npy")
# load_to_nifti(f"{RESULT_DIR}/x_ddpm_single_999.npy")
# npy_file_path = f"{RESULT_DIR}/x2_ddpm_drop_context_400.npy"
# load_to_nifti(npy_file_path)

valid_loader = DataLoader(ACDCDataset(data_dir=DATA_DIR, split="tst"), batch_size=3, shuffle=False, num_workers=1)
image_size = (32, 128, 128)
num_frames = 3

n_T = 1000  # 500
n_feat = 128 # 128 ok, 256 better (but slower)

patch_size = (8, 32, 32)

nn_model = ContextUnet(in_channels=1, n_feat=n_feat, in_shape=(1, *image_size) , num_frames=num_frames) # num frames only for ddpm single
vivit_model = ViViT(image_size, patch_size, num_frames)

# vivit_model.load_state_dict(torch.load(MODEL_PATH))
# vivit_model.to(device)
# ddpm = DDPM(vivit_model=vivit_model, nn_model=nn_model,
#              betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
ddpm = SingleDDPM(nn_model=nn_model,
             betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)

model_path_no_ep = f'{RESULT_DIR}/ddpm_single_ep'
epochs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
with open('local_config.yml', 'r') as f:
    local_user_config = yaml.safe_load(f)
project = local_user_config['project']
entity = local_user_config['entity']
wandb.init(project, entity)
for ep in epochs:
    model_path = f'{model_path_no_ep}{ep}.pth'
    losses = evaluate_sampling(ddpm, model_path, valid_loader)
    wandb.log({'epoch': ep, 'val_sampling_loss_avg': np.mean(losses), 'val_sampling_loss_std': np.std(losses), 'val_sampling_loss_list': losses})

# ddpm.load_state_dict(torch.load(MODEL_PATH))
# ddpm.to(device)
# guide_w = 4.0
# 
# val_iter = iter(valid_loader)
# x, x_prev = next(val_iter)
# # np.save(f"{RESULT_DIR}/x_val_example.npy", x)
# # plot_slice_from_npy(f"{RESULT_DIR}/x_val_example.npy")
# # load_to_nifti(f"{RESULT_DIR}/x_val_example.npy")
# # with torch.no_grad():
# #     x_prev = x_prev.to(device)
# #     x_gen = ddpm.vivit_model(x_prev)
# #     np.save(f"{RESULT_DIR}/x2_pretrained_combined_context_499.npy", x_gen.cpu())
# #     plot_slice_from_npy(f"{RESULT_DIR}/x2_pretrained_combined_context_499.npy")
# #     load_to_nifti(f"{RESULT_DIR}/x2_pretrained_combined_context_499.npy")
# 
# x, x_prev = next(val_iter)
# 
# with torch.no_grad(): 
#     x_prev = x_prev.to(device)
#     # x_gen = vivit_model(x_prev)
#     # np.save(f"{RESULT_DIR}/x_vivit_400.npy", x_gen.cpu())
#     # plot_slice_from_npy(f"{RESULT_DIR}/x_vivit_400.npy")
#     # load_to_nifti(f"{RESULT_DIR}/x_vivit_400.npy")
#     # ddpm.to(device)
#     x_gen, x_gen_store = ddpm.sample(x_prev, device, guide_w=guide_w)
#     print(x_gen.shape)
#     np.save(f"{RESULT_DIR}/x_g4_pretrained_combined_499.npy", x_gen.cpu())
#     plot_slice_from_npy(f"{RESULT_DIR}/x_g4_pretrained_combined_499.npy")
#     load_to_nifti(f"{RESULT_DIR}/x_g4_pretrained_combined_499.npy")
# 
# x, x_prev = next(val_iter)
# compare_guide_w_samples(x_prev, ddpm, 'x2_ddpm_single_999_guides.png')
