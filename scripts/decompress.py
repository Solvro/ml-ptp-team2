import gzip
import shutil
from pathlib import Path
import os

def copy_gz(filename, src_dir, dest_dir):
    with gzip.open(src_dir / filename, 'rb') as f_in:
        with open(dest_dir / filename[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


src_dir = Path('./data/real_nii_gz')
dest_dir = Path('./data/real_nii')

os.makedirs(dest_dir, exist_ok=True)

for file in os.listdir(src_dir):
    print(str(src_dir / file))
    copy_gz(file, src_dir, dest_dir)