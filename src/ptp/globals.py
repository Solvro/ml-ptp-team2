import os
from pathlib import Path

DATA_DIR = Path('../data')
TARGET_DATA_DIR = DATA_DIR / 'generated_part1_nii_gz'
MODELS_DIR = DATA_DIR / 'models'
TRAINING_DIR = DATA_DIR / 'training'
os.makedirs(TRAINING_DIR, exist_ok=True)
