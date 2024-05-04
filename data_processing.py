import os
from typing import List
import yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import nibabel as nib
import numpy as np
import scipy.ndimage


DATA_DIR = "/media/7tb_encrypted/maltes_project"
PROJECT_DIR = "/media/7tb_encrypted/adriannas_project"
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
REGISTERED_DIR= os.path.join(DATA_DIR, "anon_images_aligned")
MEDIAN_DIR = os.path.join(DATA_DIR, "median_images")
IMG_SHAPE = (384, 384, 64)
TARGET_SHAPE = (128, 128, 32)
ZOOM = [1/3, 1/3, 0.5] 
TRAINDATA_RATIO = 0.90

with open('./outliers.yaml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    outliers_list = yaml_data['files']

def extract_id(file_name: str) -> int:
    """
    Extracts id from file name (cbct<ID>_<TIMESTEP>.nii.gz)
    """
    start_index = file_name.find("cbct")
    end_index = file_name.find("_", start_index)
    # Extract the substring between 'cbct' and '_'
    substring = file_name[start_index + len("cbct") : end_index]
    return int(substring)


def extract_timestep(input_string: str) -> int:
    """
    Extracts timestep from file name (cbct<ID>_<TIMESTEP>.nii.gz)
    """
    start_index = input_string.find("_")
    end_index = input_string.find(".nii", start_index)
    # Extract the substring between '_' and '.nii'
    substring = input_string[start_index + len("_") : end_index]
    return int(substring)


def get_sorted_unique_list_of_files(folder_path: str) -> List[str]:
    """
    This function gets all cbct files from folder and sorts them by id.
    In case there are 2 files with the same id, it takes the one with bigger timestep.
    """
    cbct_files = [f for f in os.listdir(folder_path) if f.startswith("cbct") and f.endswith(".nii.gz")]
    sorted_filenames = sorted(cbct_files, key=extract_id)
    unique_files = {}
    for filename in sorted_filenames:
        idx = extract_id(filename)
        timestep = extract_timestep(filename)
        if idx not in unique_files:
            unique_files[idx] = filename
        else:
            if timestep > extract_timestep(unique_files[idx]):
                unique_files[idx] = filename
    return list(unique_files.values())


def get_correct_files_in_folder(folder_path: str) -> List[str]:
    """
    Returns a list of files in a folder, which has increasing timesteps
    """
    correct_files = []
    cbct_files = get_sorted_unique_list_of_files(folder_path)
    cur_timestep = -10e10
    for file_name in cbct_files:
        file_timestep = extract_timestep(file_name)
        if cur_timestep < file_timestep:
            correct_files.append(os.path.join(folder_path, file_name))
            cur_timestep = file_timestep
    return correct_files


def get_patient_tensor(patient_number: str):
    # median_file = f'{MEDIAN_DIR}/median_image_p{patient_number}.nii'
    # median_img = load_img(median_file)
    # if median_img.shape != IMG_SHAPE:
      #  print('Patient', patient_number, 'has wrong shape')
       # return None 
    patient_dir = os.path.join(REGISTERED_DIR, patient_number)
    patient_folders = [os.path.join(patient_dir, f) for f in os.listdir(patient_dir)]
    patient_files = []
    for folder in patient_folders:
        folder_correct_files = get_correct_files_in_folder(folder)
        len_before = len(folder_correct_files)
        folder_correct_files = [f for f in folder_correct_files if f not in outliers_list]
        len_after = len(folder_correct_files)
        if len_after != len_before:
            print(patient_number, 'removed outliers')
        patient_files.extend(folder_correct_files)
    if len(patient_files) < 20:
        print('Patient', patient_number, 'has less than 20 imgs')
        return None 
    selected_frames_indices = [0, 4, 9, -1]
    selected_images = [patient_files[i] for i in selected_frames_indices]   # chosing 1, 5, 10 and the last img 
    imgs = [load_img(file) for file in selected_images]  # loading images 
    orginal_shape = imgs[0].shape
    zoom = (orginal_shape[0]/ TARGET_SHAPE[0], orginal_shape[1]/ TARGET_SHAPE[1], orginal_shape[2]/ TARGET_SHAPE[2])
    down_imgs = [scipy.ndimage.zoom(img, zoom, order=1) for img in imgs]
    return np.array(down_imgs)
    

def load_img(img_path: str) -> np.array:
    """
    Loads NIfTI image and converts to numpy array
    """
    nii_img = nib.load(img_path)
    img_arr = nii_img.get_fdata()
    return img_arr


def validate_downscaling_with_plot(img_path, zoom):
    original = load_img(img_path)
    downscale = scipy.ndimage.zoom(original, zoom, order=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original[:, :, original.shape[2] // 2], cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    # Plot slice of downscaled image
    axes[1].imshow(downscale[:, :, downscale.shape[2] // 2], cmap='gray')
    axes[1].set_title('Downscaled')
    axes[1].axis('off')
    if type(zoom) == list: 
        file_name = f'0.5, 1/3, 1/3{PROJECT_DIR}/downscaling_{zoom[0]}_{zoom[1]}_{zoom[2]}.jpg'
    else:
        file_name = f'{PROJECT_DIR}/downscaling_{zoom}.jpg'
    plt.savefig(file_name, pad_inches=0)

def plot_slice_from_npy(npy_file):
    img = np.load(npy_file)[0][0]
    print(img.shape)
    img = np.transpose(img, (1, 2, 0))
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(img[:, :, img.shape[2] // 2], cmap='gray')
    file_name = npy_file.split('/')[-1].split('.')[0]
    plt.savefig(f'{RESULTS_DIR}/{file_name}.jpg')

def plot_slices_sequence_from_numpy(npy_file):
    sequence = np.load(npy_file)
    print(sequence.shape)
    sequence = np.transpose(sequence, (1, 2, 0, 4, 5, 3))[0][0]
    file_name = npy_file.split('/')[-1].split('.')[0]
    fig, ax = plt.subplots()
    ax.axis('off')
    def update(frame):
        ax.imshow(sequence[frame][:, :, sequence[0].shape[2] // 2], cmap='gray')
    ani = FuncAnimation(fig, update, frames=sequence.shape[0], interval=200)
    ani.save(f'{RESULTS_DIR}/{file_name}.gif', writer='pillow', fps=5)

def plot_patient_gif(folder_path, zoom):
    imgs_paths = get_correct_files_in_folder(folder_path)
    imgs = [load_img(img_path) for img_path in imgs_paths]
    down_imgs = [scipy.ndimage.zoom(img, zoom, order=1) for img in imgs]
    fig, ax = plt.subplots()
    ax.axis('off')
    def update(frame):
        ax.imshow(down_imgs[frame][:, :, down_imgs[0].shape[2] // 2], cmap='gray')
    ani = FuncAnimation(fig, update, frames=len(down_imgs), interval=200)
    ani.save(f'{PROJECT_DIR}/sequence_{zoom}.gif', writer='pillow', fps=5)


def save_data():
    all_dat = []
    for patient in os.listdir(REGISTERED_DIR):
        print(patient)
        t = get_patient_tensor(patient)
        if t is not None:
            all_dat.append(t)
    data_size = len(all_dat)
    print('Number of sequences in dataset:', data_size)
    all_dat = np.array(all_dat, dtype=np.float32)
    print('Data shape', all_dat.shape)
    # min-max scaling
    all_dat -= np.amin(all_dat, axis=(2,3,4), keepdims=True)
    all_dat /= np.amax(all_dat, axis=(2,3,4), keepdims=True)
    np.random.seed(0)
    train_length = int(data_size * TRAINDATA_RATIO)
    print('train rows:', train_length)
    print('test rows:', data_size-train_length)
    seed_value = 42
    np.random.seed(seed_value)
    rand_idx = np.random.permutation(data_size)
    trn_dat = all_dat[rand_idx[:train_length]]
    tst_dat = all_dat[rand_idx[train_length:]]
    np.save(os.path.join(PROJECT_DIR, "trn_dat.npy"), trn_dat)
    np.save(os.path.join(PROJECT_DIR, "tst_dat.npy"), tst_dat)


save_data()
