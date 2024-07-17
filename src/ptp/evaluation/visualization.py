import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import torch

DIMS = ['x', 'y', 'z']


def plot_slice(nib_img, x, y, z, title):
    plt.figure(figsize=(12, 4), constrained_layout=True)
    plt.suptitle(title)
    slices = [np.s_[x, :, :], np.s_[:, y, :], np.s_[:, :, z]]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(nib_img.get_fdata()[slices[i]], cmap='Greys')
        plt.title(f'Slice along {DIMS[i]}')


def visualize_volumes(data_dict, idx, keys, labels, is_color_channel=False):
    nonzero_mask = torch.nonzero(data_dict['mask'].type(torch.bool))
    random_slice = nonzero_mask[np.random.randint(0, nonzero_mask.shape[0])]
    if is_color_channel:
        _, _, x, y, z = random_slice
        slices = [np.s_[0, 0, x, :, :], np.s_[0, 0, :, y, :], np.s_[0, 0, :, :, z]]
    else:
        _, x, y, z = random_slice
        slices = [np.s_[0, x, :, :], np.s_[0, :, y, :], np.s_[0, :, :, z]]

    for i, (key, label) in enumerate(zip(keys, labels)):
        plt.figure(figsize=(12, 4), constrained_layout=True)
        plt.suptitle(f'{label} {idx}')
        for j in range(3):
            plt.subplot(1, 3, j + 1)
            plt.title(f'Slice along {DIMS[j]}')
            plt.imshow(data_dict[key].detach()[slices[j]], cmap='Greys')


def visualize_slices(data_dict, idx, keys, labels, is_color_channel=False):
    if is_color_channel:
        slice_ = np.s_[0, 0, :, :]
    else:
        slice_ = np.s_[:, :]

    plt.title(f'Slice view')
    for i, (key, label) in enumerate(zip(keys, labels)):
        plt.figure(figsize=(12, 4), constrained_layout=True)
        plt.subplot(1, 3, i + 1)
        plt.suptitle(f'{label}')
        plt.imshow(data_dict[key].detach()[slice_], cmap='Greys')
