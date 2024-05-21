import numpy as np

from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import rescale, rotate
from skimage.util import crop

from notebooks.utils.transformations import Transformation, image_to_spectrum, spectrum_to_image, spectrum_to_plot


def plot_gray(image):
    if image.dtype == 'complex128':
        image = spectrum_to_plot(image)
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(image, cmap='gray')
    plt.grid(None)


def plot_image_and_spectrum(image):
    f_size = 10
    spectrum = image_to_spectrum(image)
    fig, ax = plt.subplots(1, 2, figsize=(f_size, f_size))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Image')
    ax[1].imshow(spectrum_to_plot(spectrum), cmap='gray')
    ax[1].set_title('Spectrum')
    for axis in ax:
        axis.grid(None)
    plt.show()


def plot_spectrum_transformations(image, c_map='gray', *spectrum_transformations):
    spectrum_transformations = [Transformation('default', lambda x: x)] + list(spectrum_transformations)
    f_size = 8
    spectrum = image_to_spectrum(image)
    fig, ax = plt.subplots(
        len(spectrum_transformations), 2, figsize=(f_size, f_size / 2 * len(spectrum_transformations))
    )
    if len(spectrum_transformations) == 1:
        ax = np.expand_dims(ax, axis=0)
    for i, transformation in enumerate(spectrum_transformations):
        current_spectrum = transformation.fun(spectrum)
        ax[i, 0].imshow(spectrum_to_image(current_spectrum), cmap=c_map)
        ax[i, 0].set_title(f'{transformation.name} image')
        ax[i, 0].grid(None)
        ax[i, 1].imshow(spectrum_to_plot(current_spectrum), cmap='gray')
        ax[i, 1].set_title(f'{transformation.name} spectrum')
        ax[i, 1].grid(None)
    plt.show()


def plot_spectrum_transformations_compact(image, c_map='gray', *spectrum_transformations):
    spectrum_transformations = [Transformation('default', lambda x: x)] + list(spectrum_transformations)
    f_size = 8
    spectrum = image_to_spectrum(image)
    fig, ax = plt.subplots(
        2, len(spectrum_transformations), figsize=(f_size / 2 * len(spectrum_transformations), f_size)
    )
    if len(spectrum_transformations) == 1:
        ax = np.expand_dims(ax, axis=0)
    for i, transformation in enumerate(spectrum_transformations):
        current_spectrum = transformation.fun(spectrum)
        ax[0, i].imshow(spectrum_to_image(current_spectrum), cmap=c_map)
        ax[0, i].set_title(f'{transformation.name} image')
        ax[0, i].grid(None)
        ax[1, i].imshow(spectrum_to_plot(current_spectrum), cmap='gray')
        ax[1, i].set_title(f'{transformation.name} spectrum')
        ax[1, i].grid(None)
    plt.show()


def play_with_source_image(image):
    image = rgb2gray(image), 'image'
    rotated = rotate(image[0], angle=30), 'rotated'
    rescaled = rescale(image[0], scale=0.5), 'rescaled'
    rescaled_very = rescale(image[0], scale=0.1), 'rescaled_very'
    cropped = crop(image[0], crop_width=(100, 100)), 'cropped'
    rotated_cropped = crop(rotate(image[0], angle=30), crop_width=(100, 100)), 'rotated_cropped'
    transformations = [image, rotated, rescaled, rescaled_very, cropped, rotated_cropped]

    f_size = 8
    fig, ax = plt.subplots(2, len(transformations), figsize=(f_size / 2 * len(transformations), f_size))
    if len(transformations) == 1:
        ax = np.expand_dims(ax, axis=0)

    for i, curr_img in enumerate(transformations):
        img, title = curr_img
        ax[0, i].imshow(img, cmap='gray')
        ax[0, i].set_title(title)
        ax[0, i].grid(None)
        ax[1, i].imshow(spectrum_to_plot(image_to_spectrum(img)), cmap='gray')
        ax[1, i].set_title(title + ' spectrum')
        ax[1, i].grid(None)
    plt.show()
