import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from fractions import Fraction

from aspire.volume import Volume
from aspire.image import Image
from aspire.source import Simulation
from aspire.operators import FunctionFilter
from aspire.noise import CustomNoiseAdder

from center_detector.utils.convolution_utils import apply_filter
from center_detector.utils.plot_utils import save_radial_info, save_patches_with_stats_info
from center_detector.utils.general_utils import get_n_points

logger = logging.getLogger('aspire.storage')
logger.propagate = False
DATA_DIR = os.path.join(os.path.dirname(__file__), "")

reconstruction_params = {
    'img_size': 55,  # Down-sample the images/reconstruction to a desired resolution
}

filtering_params = {
    'top_n': 10,  # How many points to use for centering
    'filter_ratio': .3,  # Filter size ratio to image
    'particle_radius': 14,  # Approximate radius of a particle
    'max_val': False,  # Use max value as center
    'cut_landscape': False,  # Take only valid result of the landscape (without padding, rolling etc.)
    'circle_cut': True,  # Set values of filter outside the circle to zero
    'steepness': [.05, .15, .25, .35, .45],  # Each value represents steepness of the gaussian filter
    'start': 0,  # From what radius to save the radial data
    'end': 35,  # Up what radius to save the radial data
    'moving_avg': 1,  # Use moving averaging to smooth the radial data (relevant if larger than 1)
    'batch_size': 2048,
    'new_centers': False,
}

experimental_params = {
    'n_noisy_images': 1000,  # Number of good patches
    'n_noise_images': 1000,  # Number of outliers
    'resolution': reconstruction_params['img_size'],
    'filter_size': np.round(reconstruction_params['img_size'] * filtering_params['filter_ratio'], 0).astype(int),
    'results_path': "clustering_results/particles_vs_outliers/",
    'save_examples': False,  # Save test samples
    'use_pure_noise': True,  # Use pure noise images in outlier class
    'use_part_particles': True,  # Use partial particle images in outlier class
    'use_multi_particles': True,  # Use multiple particles images in outlier class
    'particle_shift': 0,  # allow X pixels shift from the center in random direction
    'noise_shift': .25,  # allow X percentage shift from the center in random direction of outlier images
    'noise_vars': np.geomspace(1, 1e-3, 30),
    'pad_values': 10,

}

experimental_params['add_value'] = int(experimental_params['filter_size'] / 2)
experimental_params['f_size_threshold'] = int(experimental_params['filter_size'] / 2)
experimental_params['add_value'] = experimental_params['f_size_threshold'] if filtering_params['cut_landscape'] else 0

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
v = v.downsample(experimental_params['resolution'])

empty_v = Volume(np.zeros(
    (experimental_params['resolution'], experimental_params['resolution'], experimental_params['resolution'])).astype(
    dtype))


def roll_func(imgs: np.array, range):
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
        f2 = noise_var * (x ** 2 + y ** 2) / experimental_params['resolution'] ** 2
        return (alpha * f1 + beta * f2) / 2.


    seed1 = np.random.randint(0, 1000)
    seed2 = np.random.randint(0, 1000)
    seed3 = np.random.randint(0, 1000)
    seed4 = np.random.randint(0, 1000)

    noise_adder = CustomNoiseAdder(noise_filter=FunctionFilter(noise_func))
    outliers = []

    if experimental_params['use_multi_particles']:
        glued_src = Simulation(
            L=experimental_params['resolution'],
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

        shift_range = int(experimental_params['noise_shift'] * experimental_params['resolution'])
        glued_images = roll_func(glued_images._data, (int(shift_range / 2), shift_range))
        glued_images = np.mean([glued_images[random_ind], glued_images[n_noise_images + random_ind]],
                               axis=0)
        outliers.append(glued_images)

    if experimental_params['use_pure_noise']:
        noise_src = Simulation(
            L=experimental_params['resolution'],
            n=n_noise_images,
            vols=empty_v,
            dtype=dtype,
            seed=seed3,
        )
        outliers.append(noise_src.images[:]._data)

    if experimental_params['use_part_particles']:
        part_src = Simulation(
            L=experimental_params['resolution'],
            n=n_noise_images,
            vols=v,
            dtype=dtype,
            seed=seed4,
        )
        shift_range = int(experimental_params['noise_shift'] * experimental_params['resolution'])
        part_images = roll_func(part_src.images[:]._data, (int(shift_range / 2), shift_range))
        outliers.append(part_images)

    combined_outliers = Image(np.concatenate(outliers, axis=0))
    noised_combined = noise_adder.forward(combined_outliers, np.arange(combined_outliers.shape[0]))

    noisy_src = Simulation(
        L=experimental_params['resolution'],
        n=experimental_params['n_noisy_images'],
        vols=v,
        dtype=dtype,
        seed=seed1,
    )

    single_images = noisy_src.images[:experimental_params['n_noisy_images']]._data
    rng = np.round(experimental_params['resolution'] * experimental_params['pad_values'] / np_vol.shape[0]).astype(int)
    snr = single_images[:, rng:-rng, rng:-rng].mean() / np.sqrt(noise_adder.noise_var)
    if experimental_params['particle_shift'] > 0:
        shift_range = experimental_params['particle_shift']
        single_images = roll_func(single_images, (0, shift_range), 0.0)
    single_images_clean = Image(single_images)
    single_images = noise_adder.forward(single_images_clean, np.arange(experimental_params['n_noisy_images']))
    labels = np.concatenate([np.ones(experimental_params['n_noisy_images']), np.zeros(noised_combined.shape[0])])
    combined_images = np.concatenate([single_images._data, noised_combined._data], axis=0)

    shuffled_order = np.arange(combined_outliers.shape[0] + experimental_params['n_noisy_images'])
    np.random.shuffle(shuffled_order)

    shuffled_images = combined_images[shuffled_order]
    shuffled_labels = labels[shuffled_order]

    filtered_images = []
    radial_info = []
    non_centric = []
    outlier_range = np.round(filtering_params['particle_radius']).astype(int)

    for ind, sample in enumerate(combined_images):
        filtered_sample = apply_filter(image=sample, filter_size=experimental_params['filter_size'],
                                       circle_cut=filtering_params["circle_cut"],
                                       steepness=filtering_params["steepness"])
        if filtering_params["cut_landscape"]:
            filtered_sample = filtered_sample[
                              experimental_params['f_size_threshold']:-experimental_params['f_size_threshold'],
                              experimental_params['f_size_threshold']:-experimental_params['f_size_threshold']]

        sample_points = get_n_points(filtered_sample, n=filtering_params["top_n"],
                                     max_values=filtering_params["max_val"],
                                     add_value=experimental_params['add_value'])
        centric = True
        if len(sample_points[0]) > 1:
            dbscan_model = DBSCAN(eps=filtering_params['particle_radius'] / 2,
                                  min_samples=round(len(sample_points[0]) / 3))
            clustering_res = dbscan_model.fit_predict(np.vstack(sample_points).T)
            uniques, counts = np.unique(clustering_res, return_counts=True)
            if (len(uniques) == 1) and uniques[0] == -1:
                sample_points = (sample_points[0][:1], sample_points[1][:1])
                centric = False
            else:
                centers = []
                for un in uniques:
                    if un == -1:
                        continue
                    p = (np.array([sample_points[0][clustering_res == un].mean().round().astype(int)]),
                         np.array([sample_points[1][clustering_res == un].mean().round().astype(int)]))
                    centers.append(p)
                if len(centers) == 1:
                    sample_points = centers[0]
                else:
                    majority_class = uniques[uniques != -1][counts[uniques != -1].argmax()]
                    sample_points = (
                        np.array([sample_points[0][clustering_res == majority_class].mean().round().astype(int)]),
                        np.array([sample_points[1][clustering_res == majority_class].mean().round().astype(int)]))

        if (sample_points[0][0] < outlier_range) | (sample_points[0][0] > sample.shape[-1] - outlier_range) | (
                sample_points[1][0] < outlier_range) | (sample_points[1][0] > sample.shape[-1] - outlier_range):
            centric = False

        if not centric:
            non_centric.append(ind)
        rad_info = save_radial_info(patches=[sample], labels=['Sample'],
                                    filter_size=experimental_params['filter_size'],
                                    points=[sample_points], path='', plot=False, return_vals=True)
        r_mean = np.convolve(rad_info['Sample']['radial_mean'][filtering_params["start"]:filtering_params["end"]],
                             np.ones(filtering_params["moving_avg"]) / filtering_params["moving_avg"], 'same')
        r_mean = r_mean / r_mean.max()
        radial_info.append(r_mean)
        radial_info.append(r_mean)

        filtered_images.append(filtered_sample.flatten())

    res = {"filtered_images": filtered_images, "radial_info": radial_info, "non_centric": non_centric}

    cluster_res = []
    centric_res = np.ones(len(labels)).astype(bool)
    centric_res[non_centric] = False
    cluster_res.append(centric_res)
    cluster_types = ['centric']
    accs = []
    trps = []
    trns = []

    for result in cluster_res:
        info = result == labels
        accs.append(np.sum(info) / len(info))
        trps.append(info[labels == 1].sum() / experimental_params['n_noisy_images'])
        trns.append(info[labels == 0].sum() / (info.shape[0] - experimental_params['n_noisy_images']))

    if os.path.exists(
            experimental_params["results_path"] + f'combined shift {experimental_params["particle_shift"]}.csv'):
        total_res = pd.read_csv(
            experimental_params["results_path"] + f'combined shift {experimental_params["particle_shift"]}.csv')
    else:
        total_res = pd.DataFrame(columns=['SNR', 'acc', 'trp', 'trn'])
    total_res = total_res.append(pd.DataFrame.from_dict(
        {'SNR': np.round(snr, 4), 'acc': np.round(accs, 3), 'trp': np.round(trps, 3), 'trn': np.round(trns, 3)}))
    total_res.to_csv(
        experimental_params["results_path"] + f'combined shift {experimental_params["particle_shift"]}.csv',
        index=False)

    if experimental_params['save_examples']:
        # Save examples
        particle_ind = np.random.randint(0, experimental_params['n_noisy_images'])
        noise_ind = np.random.randint(0, noised_combined.shape[0])
        single, glued = single_images[particle_ind]._data[0], noised_combined[noise_ind]._data[0]
        filtered_particle = apply_filter(image=single, filter_size=experimental_params['filter_size'],
                                         circle_cut=filtering_params["circle_cut"],
                                         steepness=filtering_params["steepness"])
        filtered_noise = apply_filter(image=glued, filter_size=experimental_params['filter_size'],
                                      circle_cut=filtering_params["circle_cut"],
                                      steepness=filtering_params["steepness"])

        particle_points = get_n_points(filtered_particle, n=filtering_params["top_n"],
                                       max_values=filtering_params["max_val"],
                                       add_value=experimental_params['add_value'])
        noise_points = get_n_points(filtered_noise, n=filtering_params["top_n"], max_values=filtering_params["max_val"],
                                    add_value=experimental_params['add_value'])

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
                                    filter_size=experimental_params['filter_size'],
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
results['STD'] = results[['acc', 'trp', 'trn']].astype(float).std(axis=1)
plt.plot(results['SNR'], results['acc'] * 100, label='Accuracy')
plt.plot(results['SNR'], results['trp'] * 100, label='True Positive')
plt.plot(results['SNR'], results['trn'] * 100, label='True negative')
plt.vlines(results.iloc[results['STD'].argsort(0)[:2]]['SNR'].mean(), 0, 100, linestyles='--', colors='red',
           label='Equality')
plt.xscale('logit')
locs, labels = plt.xticks()
plt.xlim(max(results['SNR']), min(results['SNR']))
plt.ylim(0, 100)
plt.xlabel('SNR')
plt.ylabel('Percentage')
plt.legend()
plt.xticks(l, [str(Fraction(i).limit_denominator()) for i in l])
plt.savefig(experimental_params["results_path"] + f'combined shift {experimental_params["particle_shift"]}.png')
plt.close()
