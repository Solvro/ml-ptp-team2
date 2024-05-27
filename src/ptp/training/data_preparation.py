import os

from src.ptp.globals import TARGET_DATA_DIR


def prepare_files_dirs(target_data_dir, one_sample_mode=False):
    targets = sorted(os.listdir(target_data_dir))

    if one_sample_mode:
        train_targets = targets[:1]
        val_targets = targets[:1]
    else:
        train_targets = targets[:20]
        val_targets = targets[20:]

    train_dict = [
        {'target': target_data_dir / target_name} for target_name in train_targets
    ]

    val_dict = [
        {'target': target_data_dir / target_name} for target_name in val_targets
    ]

    return train_dict, val_dict
