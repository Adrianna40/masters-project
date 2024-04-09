import os

# Path to the directory containing patient folders
root_dir = '/media/7tb_encrypted/julians_project/anon_images_updated/'

# Function to count folders within each patient's folder
def count_folders_per_patient(root_dir):
    patients = os.listdir(root_dir)
    for patient_id in patients:
        patient_folder = os.path.join(root_dir, patient_id)
        if os.path.isdir(patient_folder):
            num_folders = len([name for name in os.listdir(patient_folder) if os.path.isdir(os.path.join(patient_folder, name))])
            if num_folders > 1: 
                print(f"Patient {patient_id}: {num_folders} folders")

# Call the function with the root directory path
count_folders_per_patient(root_dir)