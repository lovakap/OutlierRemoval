from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import os

from center_detector.utils.general_utils import get_stats_info, radial_info_with_n_centers, update_range


def save_patches_with_stats_info(patches: List[np.ndarray], labels: List[str], path: str, sample_label: str,
                                 points: List[Tuple[List[int], List[int]]] = None, info=True, images=True, plot=False):
    """
    Saves patches with stats info
    :param patches:
    :param labels:
    :param path:
    :param sample_label:
    :param info:
    :param plot:
    :return:
    """
    os.makedirs(path, exist_ok=True)
    fig = plt.figure(figsize=(4 * len(patches), 6), dpi=80)
    fig.suptitle(f'{sample_label}')
    for i, patch in enumerate(patches):
        ax = fig.add_subplot(1, len(patches), i + 1)
        if images:
            im = ax.imshow(patch, cmap='gray')
            if points is not None:
                ax.scatter(points[i][1], points[i][0])
            ax.axis('off')
        else:
            im = ax.plot(patch)
        if info:
            ax.title.set_text(f'{labels[i]}\n' + get_stats_info(patch))
        # fig.colorbar(im)
    # plt.savefig(path + f'original SNR {float("{:.6f}".format(snr))}.png', edgecolor='none')
    plt.savefig(path + f'{sample_label}.png', edgecolor='none')
    if plot:
        plt.show()
    plt.close()


def save_filtered_patches_with_stats_info(patches: List[np.ndarray], labels: List[str], filter_size: int,
                                          true_centers: List[Tuple[List[int], List[int]]],
                                          points: List[Tuple[List[int], List[int]]], path: str, sample_label: str = "",
                                          mark_points: bool = True, info=True, plot=False):
    """
    Saves given filtered patches with stats info
    :param patches:
    :param labels:
    :param filter_size:
    :param true_centers:
    :param points:
    :param path:
    :param sample_label:
    :param mark_points:
    :param info:
    :param plot:
    :return:
    """
    os.makedirs(path, exist_ok=True)
    fig = plt.figure(figsize=(4 * len(patches), 8), dpi=80)
    fig.suptitle(f'filter size - {filter_size} with top {len(points[0][0])} points')

    for i, patch in enumerate(patches):
        ax = fig.add_subplot(1, len(patches), i + 1)
        im = ax.imshow(patch)
        if mark_points:
            for j in range(len(points[i][0])):
                ax.scatter(points[i][1][j], points[i][0][j], color='r')
            ax.scatter(true_centers[i][1], true_centers[i][0], color='purple')
        ax.axis('off')
        if info:
            ax.title.set_text(f'{labels[i]}\n' + get_stats_info(patch, points[i]))
        fig.colorbar(im)

    plt.savefig(path + f'filtered patch {sample_label}.png', edgecolor='none')
    if plot:
        plt.show()
    plt.close()


def save_radial_info(patches: List[np.ndarray], labels: List[str], filter_size: int,
                     points: List[Tuple[List[int], List[int]]], path: str, plot=True, return_vals=False,
                     sample_label: str = "", noise_mean=None, noise_var=None):
    """
    saves/returns radial info of given patches
    :param patches:
    :param labels:
    :param filter_size:
    :param points:
    :param path:
    :param plot:
    :param sample_label:
    :param return_vals:
    :param noise_mean:
    :param noise_var:
    :return:
    """

    radial_info = {}
    shifted_patches = {}

    mean_range = (None, None)
    var_range = (None, None)

    for i, patch in enumerate(patches):
        rd_mean, shifted_by_mean = radial_info_with_n_centers(image=patch, points=points[i], func_type='mean')
        rd_var, shifted_by_var = radial_info_with_n_centers(image=patch, points=points[i], func_type='var')

        mean_range = update_range(rd_mean, mean_range)
        var_range = update_range(rd_var, var_range)

        radial_info[labels[i]] = {'radial_mean': rd_mean, 'radial_var': rd_var}
        shifted_patches[labels[i]] = {'radial_mean': shifted_by_mean, 'radial_var': shifted_by_var}

    if return_vals:
        return radial_info

    # os.makedirs(path, exist_ok=True)
    fig = plt.figure(figsize=(8, 8), dpi=80)
    fig.suptitle(f'filter size - {filter_size}, with mean of top {len(points[0][0])} points')

    ax1 = fig.add_subplot(2, 1, 1)
    for i in range(len(patches)):
        ax1.plot(radial_info[labels[i]]['radial_mean'], label=labels[i])
    if noise_mean is not None:
        ax1.plot(noise_mean, label='Noise Mean')
    ax1.legend()
    ax1.set_ylim(mean_range[0], mean_range[1])
    ax1.set(xticklabels=[])
    ax1.tick_params(bottom=False)
    ax1.title.set_text(f'Mean')

    ax2 = fig.add_subplot(2, 1, 2)
    for i in range(len(patches)):
        ax2.plot(radial_info[labels[i]]['radial_var'], label=labels[i])
    if noise_var is not None:
        ax2.plot(noise_var, label='Noise Var')
    ax2.legend()
    ax2.set_ylim(var_range[0], var_range[1])
    # ax2.set(xticklabels=[])
    # ax2.tick_params(bottom=False)
    ax2.title.set_text(f'Var')

    plt.savefig(path + f'radial mean {sample_label}.png', edgecolor='none')
    if plot:
        plt.show()
    plt.close()


def save_plot_with_counts(labels, data, i, path):
    os.makedirs(path, exist_ok=True)
    min_val = np.min(data, axis=0).mean()
    max_val = np.max(data, axis=0).mean() * 1.3
    if min_val < 0:
        min_val *= 1.3
    else:
        min_val = 0
    nd_data = np.vstack(data)
    class_0 = nd_data[np.array(labels, dtype=bool)]
    class_1 = nd_data[~np.array(labels, dtype=bool)]
    plt.plot(class_0.mean(axis=0), label=f"class 0 - {class_0.shape[0]}")
    plt.plot(class_1.mean(axis=0), label=f"class 1 - {class_1.shape[0]}")
    plt.ylim(min_val, max_val)
    plt.legend()
    plt.savefig(f'{path}plot_{i}.png')
    plt.close()
