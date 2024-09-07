import os
import argparse
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from functools import partial
from pathos.pools import ProcessPool
from aspire.volume import Volume
from aspire.image import Image
from aspire.source import Simulation
from aspire.operators import FunctionFilter
from aspire.noise import CustomNoiseAdder

from center_detector.utils.convolution_utils import apply_filter
from center_detector.utils.plot_utils import save_radial_info, save_patches_with_stats_info
from center_detector.utils.general_utils import get_n_points
from image_filtering import run_experimental, extract_info

import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

logger = logging.getLogger('aspire.storage')
logger.propagate = False
DATA_DIR = os.path.join(os.path.dirname(__file__), "")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process filtering and experimental parameters.")

    # Filtering params
    parser.add_argument('--patch_size', type=int, default=33, help="Down-sample the images/reconstruction to a desired resolution")
    parser.add_argument('--top_n', type=int, default=10, help="How many points to use for centering")
    parser.add_argument('--filter_ratio', type=float, default=0.3, help="Filter size ratio to image")
    parser.add_argument('--particle_radius', type=int, default=5, help="Approximate radius of a particle")
    parser.add_argument('--max_val', action='store_true', help="Use max value as center")
    parser.add_argument('--cut_landscape', action='store_true', help="Take only valid result of the landscape")
    parser.add_argument('--circle_cut', action='store_true', help="Set values of filter outside the circle to zero")
    parser.add_argument('--steepness', nargs='+', type=float, default=[0.05, 0.15, 0.25, 0.35, 0.45], help="Each value represents steepness of the gaussian filter")
    parser.add_argument('--start', type=int, default=0, help="From what radius to save the radial data")
    parser.add_argument('--end', type=int, default=35, help="Up what radius to save the radial data")
    parser.add_argument('--moving_avg', type=int, default=1, help="Use moving averaging to smooth the radial data")
    parser.add_argument('--batch_size', type=int, default=2048, help="Batch size")
    parser.add_argument('--new_centers', action='store_true', help="Generate new centers for particles")

    # Experimental params
    parser.add_argument('--n_noisy_images', type=int, default=1000, help="Number of good patches")
    parser.add_argument('--n_noise_images', type=int, default=1000, help="Number of outliers")
    parser.add_argument('--n_batches', type=int, default=4, help="Number of batches")
    parser.add_argument('--results_path', type=str, default="clustering_results/test/", help="Path to save results")
    parser.add_argument('--save_examples', action='store_true', help="Save test samples")
    parser.add_argument('--use_pure_noise', action='store_true', help="Use pure noise images in outlier class")
    parser.add_argument('--use_part_particles', action='store_true', help="Use partial particle images in outlier class")
    parser.add_argument('--use_multi_particles', action='store_true', help="Use multiple particles images in outlier class")
    parser.add_argument('--particle_shift', type=int, default=0, help="Allow X pixels shift from the center in random direction")
    parser.add_argument('--noise_shift', type=float, default=0.66, help="Allow X percentage shift from the center in random direction of outlier images")
    parser.add_argument('--noise_vars', nargs='+', type=float, default=np.geomspace(1e-1, 1e-4, 5).tolist(), help="Noise variance values for outliers")
    parser.add_argument('--pad_values', type=int, default=10, help="Padding value")

    args = parser.parse_args()
    return args


def execute_experiment(args):
    filtering_params = {
        'patch_size': args.patch_size,
        'top_n': args.top_n,
        'filter_ratio': args.filter_ratio,
        'particle_radius': args.particle_radius,
        'max_val': args.max_val,
        'cut_landscape': args.cut_landscape,
        'circle_cut': args.circle_cut,
        'steepness': args.steepness,
        'start': args.start,
        'end': args.end,
        'moving_avg': args.moving_avg,
        'batch_size': args.batch_size,
        'new_centers': args.new_centers,
    }

    experimental_params = {
        'n_noisy_images': args.n_noisy_images,
        'n_noise_images': args.n_noise_images,
        'n_batches': args.n_batches,
        'results_path': args.results_path,
        'save_examples': args.save_examples,
        'use_pure_noise': args.use_pure_noise,
        'use_part_particles': args.use_part_particles,
        'use_multi_particles': args.use_multi_particles,
        'particle_shift': args.particle_shift,
        'noise_shift': args.noise_shift,
        'noise_vars': args.noise_vars,
        'pad_values': args.pad_values,

    }
    filtering_params['filter_size'] = np.round(filtering_params['patch_size'] * filtering_params['filter_ratio'], 0).astype(
        int)
    filtering_params['add_value'] = int(filtering_params['filter_size'] / 2)
    filtering_params['f_size_threshold'] = int(filtering_params['filter_size'] / 2)
    filtering_params['add_value'] = filtering_params['f_size_threshold'] if filtering_params['cut_landscape'] else 0

    os.makedirs(experimental_params['results_path'], exist_ok=True)
    if experimental_params['save_examples']:
        os.makedirs(experimental_params['results_path'] + '/examples', exist_ok=True)

    dtype = np.float64

    n_split = 0
    if experimental_params['use_pure_noise']:
        n_split += 1
    if experimental_params['use_part_particles']:
        n_split += 1
    if experimental_params['use_multi_particles']:
        n_split += 1

    n_noise_images = int(experimental_params['n_noise_images'] / n_split)

    str_labels = ['Single Object', 'Outliers']

    # Create some projections
    np_vol = np.pad(np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")).astype(dtype),
                    ((experimental_params['pad_values'], experimental_params['pad_values']),
                     (experimental_params['pad_values'], experimental_params['pad_values']),
                     (experimental_params['pad_values'], experimental_params['pad_values'])), 'edge')

    v = Volume(np_vol / np_vol.max())
    filtering_params['extraction_size'] = v.shape[1]
    empty_v = Volume(np.zeros(v.shape).astype(dtype))


    def roll_func(imgs: np.array, range, delta=0.3):
        # roll_range = (range[0], int(range[1] * (1 + delta)))
        roll_range = (range[0], int(range[1] * 1))
        rolls = np.random.choice(np.unique(np.concatenate([-np.flip(np.arange(*roll_range)), np.arange(*roll_range)])),
                                 (imgs.shape[0], 2))
        rolled_imgs = np.array([
            np.roll(np.pad(im, ((range[1], range[1]), (range[1], range[1])), 'constant', constant_values=0), rolls[i],
                    axis=(0, 1))[range[1]:-range[1], range[1]:-range[1]] for i, im in enumerate(imgs)])
        return rolled_imgs


    def calc_std(points):
        stacked = np.vstack(points).T
        return np.sqrt(np.sum((stacked - np.mean(stacked, axis=0)) ** 2, axis=1).mean())


    for noise_var in experimental_params['noise_vars']:
        def noise_func(x, y):
            alpha = 1
            beta = 1
            # White
            f1 = noise_var
            # Violet-ish
            f2 = noise_var * (x ** 2 + y ** 2) / filtering_params['extraction_size'] ** 2
            return (alpha * f1 + beta * f2) / 2.

        seed1 = np.random.randint(0, 1000)
        seed2 = np.random.randint(0, 1000)
        seed4 = np.random.randint(0, 1000)
        noise_seed = np.random.randint(0, 1000)
        noise_adder = CustomNoiseAdder(noise_filter=FunctionFilter(noise_func), seed=noise_seed)
        outliers = []

        if experimental_params['use_multi_particles']:
            glued_src = Simulation(
                L=filtering_params['extraction_size'],
                n=2 * n_noise_images,
                vols=v,
                dtype=dtype,
                seed=seed2,
            )
            random_ind = np.arange(0, n_noise_images)
            np.random.shuffle(random_ind)
            random_ind2 = np.arange(0, n_noise_images)
            np.random.shuffle(random_ind2)

            glued_images = glued_src.images[0: 2 * n_noise_images]

            shift_range = int(experimental_params['noise_shift'] * filtering_params['extraction_size'])
            glued_images = roll_func(glued_images._data, (int(filtering_params['extraction_size'] * 0.1), shift_range))
            glued_images = np.mean([glued_images[random_ind], glued_images[n_noise_images + random_ind]],
                                   axis=0)
            outliers.append(glued_images)

        if experimental_params['use_pure_noise']:
            noise_src = Simulation(
                L=filtering_params['extraction_size'],
                n=n_noise_images,
                vols=empty_v,
                dtype=dtype,
                seed=noise_seed,
            )
            outliers.append(noise_src.images[:]._data)

        if experimental_params['use_part_particles']:
            part_src = Simulation(
                L=filtering_params['extraction_size'],
                n=n_noise_images,
                vols=v,
                dtype=dtype,
                seed=seed4,
            )
            shift_range = int(experimental_params['noise_shift'] * filtering_params['extraction_size'])
            part_images = roll_func(part_src.images[:]._data,
                                    (int(filtering_params['extraction_size'] * 0.25), shift_range))
            outliers.append(part_images)

        combined_outliers = Image(np.concatenate(outliers, axis=0))
        noised_combined = noise_adder.forward(combined_outliers, np.arange(combined_outliers.shape[0]))

        noisy_src = Simulation(
            L=filtering_params['extraction_size'],
            n=experimental_params['n_noisy_images'],
            vols=v,
            dtype=dtype,
            seed=seed1,
        )

        single_images = noisy_src.images[:experimental_params['n_noisy_images']]._data
        rng = np.round(filtering_params['extraction_size'] * experimental_params['pad_values'] / np_vol.shape[0]).astype(
            int)
        snr = single_images[:, rng:-rng, rng:-rng].mean() / np.sqrt(noise_adder.noise_var)
        if experimental_params['particle_shift'] > 0:
            shift_range = experimental_params['particle_shift']
            single_images = roll_func(single_images, (0, shift_range), 0.0)
        single_images_clean = Image(single_images)
        single_images = noise_adder.forward(single_images_clean, np.arange(experimental_params['n_noisy_images']))
        labels = np.concatenate([np.ones(experimental_params['n_noisy_images']), np.zeros(noised_combined.shape[0])])

        combined_images = np.concatenate([single_images._data, noised_combined._data], axis=0)
        combined_images = Image(combined_images).downsample(filtering_params['patch_size'])._data

        # Run info extraction
        stack_size = int(combined_images.shape[0] / experimental_params['n_batches'])
        with ProcessPool(int(len(os.sched_getaffinity(0)) / 2)) as pool:
            cluster_res = pool.map(partial(run_experimental, args=filtering_params),
                                   [combined_images[i:i + stack_size] for i in
                                    range(0, combined_images.shape[0], stack_size)])
        cluster_res = np.concatenate(cluster_res)

        info = cluster_res == labels
        acc = np.sum(info) / info.shape[0]
        tpr = info[labels == 1].sum() / experimental_params['n_noisy_images']
        tnr = info[labels == 0].sum() / (info.shape[0] - experimental_params['n_noisy_images'])

        if os.path.exists(
                experimental_params["results_path"] + f'combined radius {filtering_params["particle_radius"]}.csv'):
            total_res = pd.read_csv(
                experimental_params["results_path"] + f'combined radius {filtering_params["particle_radius"]}.csv')
        else:
            total_res = pd.DataFrame(columns=['SNR', 'ACC', 'TPR', 'TNR'])
        total_res = pd.concat([total_res, pd.DataFrame.from_dict(
            {'SNR': [np.round(snr, 4)], 'ACC': [np.round(acc, 3)], 'TPR': [np.round(tpr, 3)], 'TNR': [np.round(tnr, 3)]})])
        total_res.to_csv(
            experimental_params["results_path"] + f'combined radius {filtering_params["particle_radius"]}.csv',
            index=False)

        if experimental_params['save_examples']:
            # Save examples
            particle_ind = np.random.randint(0, experimental_params['n_noisy_images'])
            noise_ind = np.random.randint(0, noised_combined.shape[0])
            single, glued = single_images[particle_ind]._data[0], noised_combined[noise_ind]._data[0]
            filtered_particle = apply_filter(image=single, filter_size=filtering_params['filter_size'],
                                             circle_cut=filtering_params["circle_cut"],
                                             steepness=filtering_params["steepness"])
            filtered_noise = apply_filter(image=glued, filter_size=filtering_params['filter_size'],
                                          circle_cut=filtering_params["circle_cut"],
                                          steepness=filtering_params["steepness"])

            particle_points = get_n_points(filtered_particle, n=filtering_params["top_n"],
                                           max_values=filtering_params["max_val"],
                                           add_value=filtering_params['add_value'])
            noise_points = get_n_points(filtered_noise, n=filtering_params["top_n"], max_values=filtering_params["max_val"],
                                        add_value=filtering_params['add_value'])

            save_patches_with_stats_info(patches=[filtered_particle, filtered_noise], labels=str_labels,
                                         path=experimental_params['results_path'] + '/examples/',
                                         sample_label=f"snr_{np.round(snr, 4)} noise_{noise_var}, shift {experimental_params['particle_shift']}",
                                         points=[particle_points, noise_points], )

            save_patches_with_stats_info(
                patches=[single_images_clean[particle_ind]._data[0], combined_outliers[noise_ind]._data[0]],
                labels=str_labels,
                path=experimental_params['results_path'] + '/examples/',
                sample_label=f"snr_{np.round(snr, 4)} noise_{noise_var}, shift {experimental_params['particle_shift']}_clean")

            save_patches_with_stats_info(patches=[single, glued], labels=str_labels,
                                         path=experimental_params['results_path'] + '/examples/',
                                         sample_label=f"snr_{np.round(snr, 4)} noise_{noise_var}, shift {experimental_params['particle_shift']}_noised")

            rad_info = save_radial_info(patches=[single, glued], labels=['particle', 'noise'],
                                        filter_size=filtering_params['filter_size'],
                                        points=[particle_points, noise_points], path='', plot=False, return_vals=True)

            particle_radial_info = np.convolve(
                rad_info['particle']['radial_mean'][filtering_params["start"]:filtering_params["end"]],
                np.ones(filtering_params["moving_avg"]) / filtering_params["moving_avg"], 'same')

            noise_radial_info = np.convolve(
                rad_info['noise']['radial_mean'][filtering_params["start"]:filtering_params["end"]],
                np.ones(filtering_params["moving_avg"]) / filtering_params["moving_avg"], 'same')

            plt.close()
            plt.plot(particle_radial_info, label='Particle')
            plt.plot(noise_radial_info, label='Noise')
            plt.legend()
            plt.ylim(-.05, .15)
            plt.savefig(experimental_params['results_path'] + '/examples/' + f"radial_info_{noise_var}.png")
            plt.close()

    l = [1 / 15, 1 / 20, 1 / 30, 1 / 50, 1 / 70, 1 / 100, 1 / 150, 1 / 250, 1 / 400]
    results = total_res
    results['STD'] = results[['ACC', 'TPR', 'TNR']].astype(float).std(axis=1)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = '18'
    plt.plot(results['SNR'], results['TPR'] * 100, label='True Positive', linewidth=3.0)
    plt.plot(results['SNR'], results['TNR'] * 100, label='True Negative', linewidth=3.0)
    plt.xscale('log')
    plt.grid(axis='y', which='both')
    locs, labels = plt.xticks()
    plt.xlim(max(results['SNR']), min(results['SNR']))
    plt.ylim(0, 100)
    plt.xlabel(r'\textbf{Noise Level}')
    plt.ylabel(r'\textbf{Percentage}')
    ax = plt.gca()
    ax.set_xticks([3e-3, 1e-2, 3e-2])
    ax.set_xticklabels(['$\\mathdefault{3\\times10^{-3}}$', '$\\mathdefault{10^{-2}}$', '$\\mathdefault{3\\times10^{-2}}$'])
    ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.savefig(experimental_params["results_path"] + f'combined radius {filtering_params["particle_radius"]}.png',
                bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    args = parse_arguments()
    execute_experiment(args)