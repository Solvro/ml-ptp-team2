import os

from src.ptp.globals import TARGET_DATA_DIR


def prepare_files_dirs():
    targets = sorted(os.listdir(TARGET_DATA_DIR))

    train_dict = [
        {'target': TARGET_DATA_DIR / target_name} for target_name in targets[:1]
    ]

    val_dict = [
        {'target': TARGET_DATA_DIR / target_name} for target_name in targets[:1]
    ]

    return train_dict, val_dict
