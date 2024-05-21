import os

from src.ptp.globals import TARGET_DATA_DIR


def prepare_files_dirs(target_data_dir):
    targets = sorted(os.listdir(target_data_dir))

    train_dict = [
        {'target': target_data_dir / target_name} for target_name in targets[:1]
    ]

    val_dict = [
        {'target': target_data_dir / target_name} for target_name in targets[:1]
    ]

    return train_dict, val_dict
