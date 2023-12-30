import numpy as np
from typing import List
from skimage.draw import disk
from scipy.signal.windows import gaussian


def circle_(kernel_size: int) -> np.array:
    """
    :param kernel_size:
    :return: image of a circle
    """
    im = np.zeros((kernel_size, kernel_size))
    cc, rr = disk((int(kernel_size / 2), int(kernel_size / 2)), int(kernel_size / 2))
    im[cc, rr] = 1
    return im


def linear_kernel(kernel_size: int, circle_cut: bool = False) -> np.array:
    """
    :param kernel_size:
    :param circle_cut:
    :return: filter with linear steepness with size kernel_size x kernel_size
    """
    if kernel_size % 2 == 0:
        kernel_size -= 1
    l_kernel = list(range(int(kernel_size/2) + 1))
    l_kernel.reverse()
    l_kernel = np.array(list(range(int(kernel_size/2))) + l_kernel[0:]).reshape(kernel_size, 1) + 1
    l_kernel = l_kernel / l_kernel.sum()
    l_kernel = np.dot(l_kernel, l_kernel.T)
    if circle_cut:
        l_kernel *= circle_(kernel_size=kernel_size)
    return l_kernel


def gaussian_kernel(kernel_size: int, circle_cut: bool = False, steepness: float = .25) -> np.array:
    """
    :param kernel_size:
    :param circle_cut:
    :param steepness:
    :return: gaussian 2D filter with given steepness with size kernel_size x kernel_size
    """
    g = gaussian(kernel_size, kernel_size * steepness).reshape(kernel_size, 1)
    g = np.dot(g, g.T)
    if circle_cut:
        g *= circle_(kernel_size=kernel_size)
    return g/g.sum()


def apply_filter(image: np.ndarray, filter_size: int, steepness: List = [.2], circle_cut: bool = False) -> np.ndarray:
    """
    applying 2d convolution on image with gaussian filters with given steepness
    :param image:
    :param filter_size:
    :param steepness:
    :param circle_cut:
    :return: landscape of Image with filters
    """
    filtered_image = image.copy()
    kernel_filter = []
    for steep in steepness:
        kernel_filter.append(-gaussian_kernel(filter_size, steepness=steep, circle_cut=circle_cut))

    filtered_image = fft_convolve_2d(filtered_image, kernel_filter)
    return filtered_image


def fft_convolve_2d(image: np.ndarray, filters: List[np.ndarray]) -> np.ndarray:
    """
    applying 2d convolution on image with given filters
    :param image:
    :param filters:
    :return: centered average landscape of all landscapes
    """
    fft_image = np.fft.fft2(image)
    fft_filter = np.fft.fft2(np.sum(filters, axis=0), s=image.shape)
    fft_out = np.multiply(fft_image, fft_filter)
    out = np.fft.ifft2(fft_out)
    roll = int(filters[0].shape[0] / 2)
    return np.roll(out.real, -roll, axis=[0, 1])
