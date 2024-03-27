import os
import shutil
import zipfile
import tempfile

# Path to the zip file containing all the models
zip_file_path = '/raid/ee-udayan/uganguly/rohit/UNIXKD/ECCV2020-SSKD-Teacher-20240327T022524Z-001.zip'

# Temporary directory to extract the zip file
temp_dir = tempfile.mkdtemp()

# Extract the .zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

# Process each .pth file in the temporary directory
for root, dirs, files in os.walk(temp_dir):
    for file in files:
        if file.endswith('.pth'):
            model_name = file[:-4]  # Remove '.pth' from the file name to get the model name
            base_dir = f'experiments/teacher_{model_name}'
            ckpt_dir = os.path.join(base_dir, 'ckpt')
            original_pth_file = os.path.join(root, file)
            new_pth_file_path = os.path.join(ckpt_dir, 'best.pth')

            # Create the directory structure if it doesn't exist
            os.makedirs(ckpt_dir, exist_ok=True)

            # Move and rename the .pth file
            shutil.move(original_pth_file, new_pth_file_path)
            print(f"{file} has been moved and renamed to {new_pth_file_path}.")

# Optionally, remove the temporary directory after processing
# Be careful with this step to avoid accidentally deleting important files
# shutil.rmtree(temp_dir)
