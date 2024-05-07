import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import os
import wandb
import yaml
import sys

sys.path.insert(0, 'SADM-modified')
from DDPM import SingleDDPM, ContextUnet
from ACDC_loader import ACDCDataset

DATA_DIR = "/media/7tb_encrypted/adriannas_project"
RESULT_DIR = "/media/7tb_encrypted/adriannas_project/results"

assert os.path.isdir(DATA_DIR), f"{DATA_DIR} is not a directory."
assert os.path.isdir(RESULT_DIR), f"{RESULT_DIR} is not a directory."

def train():
    with open('local_config.yml', 'r') as f:
        local_user_config = yaml.safe_load(f)
    project = local_user_config['project']
    entity = local_user_config['entity']
    wandb.init(project, entity)
    device = torch.device("cuda")
    n_epoch = 500
    batch_size = 3
    image_size = (32, 128, 128)
    num_frames = 3

    # DDPM hyperparameters
    n_T = 1000  # 500
    n_feat = 128 # 128 ok, 256 better (but slower)
    lrate = 5e-5

    dataset = ACDCDataset(data_dir=DATA_DIR, split="trn")
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)

    valid_loader = DataLoader(ACDCDataset(data_dir=DATA_DIR, split="tst"), batch_size=batch_size, shuffle=False, num_workers=1)
    # x_val, x_prev_val = next(iter(valid_loader))
    # x_prev_val = x_prev_val.to(device)

    nn_model = ContextUnet(in_channels=1, n_feat=n_feat, in_shape=(1, *image_size), num_frames=num_frames)
    ddpm = SingleDDPM(nn_model=nn_model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)


    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(train_loader)
        loss_epoch = []
        for x, x_prev in pbar:
            optim.zero_grad()
            x = x.to(device)
            x_prev = x_prev.to(device)
            loss = ddpm(x, x_prev)
            loss.backward()
            loss_epoch.append(loss.item())
            pbar.set_description(f"loss: {loss.item():.4f}")
            optim.step()

        ddpm.eval()
        vbar = tqdm(valid_loader)
        loss_val_epoch = []
        with torch.no_grad():
            for x_val, x_prev_val in vbar:
                x_val = x_val.to(device)
                x_prev_val = x_prev_val.to(device)
                val_loss = ddpm(x_val, x_prev_val)
                vbar.set_description(f'val loss: {val_loss.item():.4f}')
                loss_val_epoch.append(val_loss.item()) 
        train_loss = np.array(loss_epoch).mean()
        val_loss = np.array(loss_val_epoch).mean()
        print('Avg Train Loss', train_loss)
        print('Avg Val Loss', val_loss)
        wandb.log({'epoch': ep, 'train_loss': train_loss, 'val_loss': val_loss})
        if ep % 100 == 0:
            torch.save(ddpm.state_dict(), f'{RESULT_DIR}/ddpm_single_ep{ep}.pth')
    with torch.no_grad():
        x_gen = ddpm.sample(x_prev_val, device, 0.4)
        np.save(f"{RESULT_DIR}/x_ddpm_single_{ep}.npy", x_gen.cpu())
    torch.save(ddpm.state_dict(), f'{RESULT_DIR}/ddpm_single_ep{ep}.pth')


if __name__=="__main__":
    train()
