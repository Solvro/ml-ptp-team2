from pathlib import Path

import nibabel as nib
import numpy as np

"""data_dir = Path('C:\\Users\\mikol\\OneDrive\\Pulpit\\ptp_dataset')
train_generated_dir = data_dir / 'generated_part1_nii_gz'
output_dir_original_seismic = data_dir / 'generated_data'

output_dir_original_seismic.mkdir(parents=True, exist_ok=True)
file_paths = list(train_generated_dir.glob('*.nii.gz'))

for i, file_path in enumerate(file_paths):
    seismic_volume = nib.load(file_path).get_fdata()[0:256, 0:256, 400:656]

    output_file_path = output_dir_original_seismic / f'seismic_volume_{i}.nii.gz'
    nib.save(nib.Nifti1Image(seismic_volume, np.eye(4)), output_file_path)

    print(f'Generated and saved meta data for file {i + 1}/{len(file_paths)}')"""



# 128 2d

data_dir = Path('C:\\Users\\mikol\\OneDrive\\Pulpit\\ptp_dataset')
train_generated_dir = data_dir / 'generated_part1_nii_gz'
output_dir_original_seismic = data_dir / 'generated_data_2d_128'

output_dir_original_seismic.mkdir(parents=True, exist_ok=True)
file_paths = list(train_generated_dir.glob('*.nii.gz'))

for i, file_path in enumerate(file_paths):
    seismic_volume = nib.load(file_path).get_fdata()[0:128, 0:128, 600]

    output_file_path = output_dir_original_seismic / f'128_seismic_volume_{i}.nii.gz'
    nib.save(nib.Nifti1Image(seismic_volume, np.eye(4)), output_file_path)

    print(f'Generated and saved meta data for file {i + 1}/{len(file_paths)}')



    
