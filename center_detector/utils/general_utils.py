from typing import Tuple, List
import numpy as np
import pandas as pd
import os
from center_detector.utils.convolution_utils import fft_convolve_2d
from aspire.storage import StarFile
from collections import OrderedDict


def get_image_der(image: np.array) -> Tuple[np.array, np.array]:
    dx = (np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])) / 8
    dy = (np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])) / 8

    x_der = fft_convolve_2d(image, [dx])
    y_der = fft_convolve_2d(image, [dy])

    return x_der, y_der


def get_der_sum(image: np.array) -> np.array:
    Ix, Iy = get_image_der(image)
    der_sum = np.sum(np.abs(Ix)) + np.sum(np.abs(Iy))
    return der_sum


def get_n_points(patch: np.ndarray, n: int = 1, max_values: bool = True, add_value: int = 0):
    if max_values:
        points = np.unravel_index(np.argsort(patch.flatten())[-n:], patch.shape)
    else:
        points = np.unravel_index(np.argsort(patch.flatten())[:n], patch.shape)

    if add_value > 0:
        points = (points[0] + add_value, points[1] + add_value)
    return points


def logs_to_star(input_dir: str, output_dir: str):
    for file in os.listdir(input_dir):
        if file.endswith('.log'):
            with open(input_dir + file) as f:
                f = f.readlines()
            for line in f:
                if 'Final Values' in line:
                    values = [val for val in line.split(" ") if val != ""][:3]
                    values = [float(val) for val in values]
            df = pd.DataFrame([values], columns=['_rlnDefocusU', '_rlnDefocusV', '_rlnDefocusAngle'])
            df['_rlnMicrographName'] = f'{file[:3]}.mrc'
            df['_rlnSphericalAberration'] = 2.0
            df['_rlnAmplitudeContrast'] = 0.1
            df['_rlnVoltage'] = 300.
            df['_rlnDetectorPixelSize'] = 1.34
            columns_sort = ['_rlnMicrographName', '_rlnDefocusU', '_rlnDefocusV', '_rlnDefocusAngle', '_rlnSphericalAberration',
                            '_rlnAmplitudeContrast', '_rlnVoltage', '_rlnDetectorPixelSize']
            df = df[columns_sort]
            """
            Writes CTF parameters to starfile for a single micrograph.
            """
            blocks = OrderedDict()
            blocks["root"] = df
            star = StarFile(blocks=blocks)
            star.write(output_dir + f'{file[:3]}.star')


def get_measure_func(name: str):
    if name == 'var':
        return np.var
    elif name == 'mean':
        return np.mean
    else:
        raise Exception('No such function')


def radial_info_with_n_centers(image: np.ndarray, points: Tuple[List[int], List[int]], func_type: str,
                               allow_edges: bool = True):
    """
    Calculate radial info based on given function (mean/var) and centers. In case of >1 centers, returns an average info
    :param image:
    :param points:
    :param func_type:
    :param allow_edges: If false - limit the location of centers up to 25% dist from the edges
    :return:
    """
    x = points[0]
    y = points[1]
    shape = image.shape
    quad_size = int(shape[0] / 4)
    func = get_measure_func(name=func_type)
    mean_radial_mean = []
    true_shifted = None

    for point in range(len(x)):
        if allow_edges:
            x_i = x[point]
            y_i = y[point]
        else:
            x_i = np.clip(x[point], quad_size, shape[0] - quad_size)
            y_i = np.clip(y[point], quad_size, shape[1] - quad_size)
        num_of_radiuses = int(min(shape) / 2)
        shift_x = np.ceil(shape[0] / 2).astype(int) - x_i
        shift_y = np.ceil(shape[1] / 2).astype(int) - y_i
        shifted = np.roll(image, shift=(shift_x, shift_y), axis=(0, 1))
        xs, ys = np.ogrid[0:shape[0], 0:shape[1]]

        shifted = shifted - shifted.mean()
        img_nrm = image - image.mean()
        radiuses = np.hypot(xs - x_i, ys - y_i)
        rbin = (num_of_radiuses * radiuses / radiuses.max()).astype(int)
        # radial_mean = np.asarray([func(img_nrm[rbin <= i]) for i in np.arange(0, rbin.max())])
        radial_mean = np.asarray([func(img_nrm[(rbin <= i) & (rbin >= i - 1)]) for i in np.arange(0, rbin.max())])
        mean_radial_mean.append(radial_mean[:2 * quad_size - 1])

        # Save the shift by the first point
        if point == 0:
            true_shifted = shifted
    return np.mean(mean_radial_mean, axis=0), true_shifted


def _calc_patch(patch, sub_patch_size, func):
    if sub_patch_size[0] > patch.shape[0]:
        sub_patch_size = (patch.shape[0], sub_patch_size[1])

    if sub_patch_size[1] > patch.shape[1]:
        sub_patch_size = (sub_patch_size[0], patch.shape[1])

    patch_var = np.zeros((patch.shape[0] - sub_patch_size[0] + 1, patch.shape[1] - sub_patch_size[1] + 1))
    for i in range(patch_var.shape[0]):
        for j in range(patch_var.shape[1]):
            patch_var[i][j] = func(patch[i:sub_patch_size[0] + i, j:sub_patch_size[0] + j])
    return np.mean(patch_var)


def update_range(values: np.array, given_range: Tuple[float]) -> Tuple[float, float]:
    low_bound = np.min(values) if given_range[0] is None else min(np.min(values), given_range[0])
    high_bound = np.max(values) if given_range[1] is None else max(np.max(values), given_range[1])
    return low_bound, high_bound


def get_stats_info(patch: np.ndarray, points=None, filter_size: int = None, mean: bool = True, std: bool = True,
                   der_sum: bool = False,
                   return_as_dict: bool = False) -> str:
    """
    Default calculates mean, std, derivative sum of the patch and variance of given centers (in case there are more than
     one)
    :param patch:
    :param points:
    :param mean:
    :param std:
    :param der_sum:
    :param return_as_dict:
    :return:
    """
    if return_as_dict:
        info = {}
        if mean:
            info['mean'] = np.mean(patch)
        if std:
            info['std'] = _calc_patch(patch, (filter_size, filter_size), np.std)
        if der_sum:
            info['der_sum'] = get_der_sum(patch)
        if points is not None:
            if len(points[0]) > 1:
                particle_center_var = np.sum((np.squeeze(points).T - np.squeeze(points).mean(axis=1)) ** 2) / len(
                    points)
            elif len(points[0]) == 1:
                particle_center_var = 0.0
            info['center_var'] = particle_center_var
    else:
        info = ''
        if mean:
            info += f'Mean: {float("{:.6f}".format(np.mean(patch)))}\n'
        if std:
            info += f'STD: {float("{:.6f}".format(np.std(patch)))}\n'
        if der_sum:
            info += f'Der Sum: {float("{:.4f}".format(get_der_sum(patch)))}\n'
        if points is not None:
            if len(points[0]) > 1:
                particle_center_var = np.sum((np.squeeze(points).T - np.squeeze(points).mean(axis=1)) ** 2) / len(
                    points)
            elif len(points[0]) == 1:
                particle_center_var = 0.0
            info += f'Center Var: {float("{:.4f}".format(particle_center_var))}'

    return info

