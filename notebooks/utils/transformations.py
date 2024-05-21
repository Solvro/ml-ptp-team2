import math

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np


def spectrum_to_plot(spectrum):
    epsilon = 1e-10
    return np.log(abs(spectrum) + epsilon)


def image_to_spectrum(image):
    return np.fft.fftshift(np.fft.fft2(image))


def spectrum_to_image(spectrum):
    return abs(np.fft.ifft2(spectrum))


@dataclass
class Transformation:
    name: str
    fun: Callable[[np.ndarray], np.ndarray]


def get_horizontal_bar_transformation(size=5, shift=0, filler=1, crop_center=0):
    def horizontal_bar(image):
        image = deepcopy(image)
        center = len(image) // 2
        image[center - size + shift : center + size + shift, : center - crop_center] = filler
        image[center - size + shift : center + size + shift, center + crop_center :] = filler
        return image

    return Transformation(name=f'horizontal bar size {size}, filled {filler}', fun=horizontal_bar)


def get_vertical_bar_transformation(size=5, shift=0, filler=1, crop_center=0):
    def horizontal_bar(image):
        image = deepcopy(image)
        center = len(image) // 2
        image[: center - crop_center, center - size + shift : center + size + shift] = filler
        image[center + crop_center :, center - size + shift : center + size + shift] = filler
        return image

    return Transformation(name=f'vertical bar size {size}, filled {filler}', fun=horizontal_bar)


def get_rotated_bar_transformation(size=5, angle=90, crop_center=0, filler=1):
    def rotated(image):
        x = len(image) // 2
        y = len(image[0]) // 2
        a = np.tan(angle * math.pi / 180)
        image = deepcopy(image)
        for i in range(len(image)):
            for j in range(len(image[i])):
                _y = a * (i - x) + y
                if math.sqrt((x - i) ** 2 + (y - j) ** 2) > crop_center and abs(_y - j) <= size:
                    image[i][j] = filler
        return image

    return Transformation(name=f'rotated at {angle}', fun=rotated)


def get_crop_transformation(radius=10, filler=1):
    def crop(image):
        x = len(image) // 2
        y = len(image[0]) // 2
        image = deepcopy(image)
        for i in range(len(image)):
            for j in range(len(image[i])):
                if math.sqrt((x - i) ** 2 + (y - j) ** 2) > radius:
                    image[i][j] = filler
        return image

    return Transformation(name=f'crop r {radius}', fun=crop)


def get_circle_transformation(radius=10, filler=1):
    def circle(image):
        x = len(image) // 2
        y = len(image[0]) // 2
        image = deepcopy(image)
        for i in range(len(image)):
            for j in range(len(image[i])):
                if math.sqrt((x - i) ** 2 + (y - j) ** 2) <= radius:
                    image[i][j] = filler
        return image

    return Transformation(name=f'circle r {radius}', fun=circle)


def compose_transformations(*transformations):
    def composed(image):
        for transformation in transformations:
            image = transformation.fun(image)
        return image

    return Transformation(name='composed', fun=composed)
