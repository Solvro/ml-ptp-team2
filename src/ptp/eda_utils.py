import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

DIMS = ['x', 'y', 'z']

def stat_along_slice(nib_array, slice):
    mean_along_slice = np.mean([img.get_fdata()[slice] for img in nib_array])
    std_along_slice = np.std([img.get_fdata()[slice] for img in nib_array])
    return mean_along_slice, std_along_slice

def compute_stats_along_slices(nib_array, slices, coords):
    data = pd.DataFrame({i : (*stat_along_slice(nib_array, slice), coords[i]) for i, slice in enumerate(slices)})
    data = data.transpose().rename({0: 'mean', 1: 'std', 2: 'coord'}, axis=1)
    return data

def compute_stats_along_x(nib_array, x_s):
    data = compute_stats_along_slices(nib_array, [np.s_[x, :, :] for x in x_s], x_s)
    return data

def compute_stats_along_y(nib_array, y_s):
    data = compute_stats_along_slices(nib_array, [np.s_[:, y, :] for y in y_s], y_s)
    return data

def compute_stats_along_z(nib_array, z_s):
    data = compute_stats_along_slices(nib_array, [np.s_[:, :, z] for z in z_s], z_s)
    return data

def visualize_slices(nib_img, x, y, z, title):
    plt.figure(figsize=(12, 4), constrained_layout=True)
    plt.suptitle(title)
    slices = [np.s_[x, :, :], np.s_[:, y, :], np.s_[:, :, z]]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(nib_img.get_fdata()[slices[i]], cmap='Greys')
        plt.title(f'Slice along {DIMS[i]}')


def plot_mean_and_std(df, ax, title=''):
    ax2 = ax.twinx()
    ax.plot(df['coord'], df['mean'], color='limegreen', marker='o')
    ax2.plot(df['coord'], df['std'], color='pink', marker='*')
    ax.set_xlabel('coordinate')
    ax.set_ylabel('mean')
    ax2.set_ylabel('std')
    ax.set_title(title)

    return ax, ax2

def plot_dist_along_each_dimension(volumin, title):
    plt.figure(figsize=(12, 4), constrained_layout=True)
    plt.suptitle(title)
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.hist([img.header.get_data_shape()[i] for img in volumin])
        plt.title(f'Dim {DIMS[i]}')


def plot_spacial_dim_dist(nib_array, title=''):
    plt.figure(figsize=(12, 4), constrained_layout=True)
    plt.suptitle(title)
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.hist([img.header.get_data_shape()[i] for img in nib_array])
        plt.title(f'Dim {DIMS[i]}')


def plot_mean_and_std_dist(means, stds, title=''):
    plt.figure(figsize=(12, 6), constrained_layout=True)
    plt.suptitle(title)
    plt.subplot(1, 2, 1)
    plt.hist(means)
    plt.title('Mean distribution')
    plt.subplot(1, 2, 2)
    plt.title('Std distribution')
    _ = plt.hist(stds)
