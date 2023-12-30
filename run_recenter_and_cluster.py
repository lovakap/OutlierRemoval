import glob
import logging

from cluster_images import run_coord_update
from center_images import run_center_update

logger = logging.getLogger('aspire.storage')
logger.propagate = False
COORD_COLS = ['_rlnCoordinateX', '_rlnCoordinateY']

filtering_params = {
    'dataset_name': '10028',
    'extraction_size': 360,  # Patch size to crop from the micrograph
    'patch_size': 55,  # Down-sample the images/reconstruction to a desired resolution
    'pixel_size': 1.34,
    'data_dir': '10028/',  # Directory with mrc, coordinates files and CTF files (expecting CTF and coordinates to be in Relion format)
    'top_n': 10,  # How many points to use for centering
    'filter_ratio': .3,  # Filter size ratio to image
    'particle_radius': 14,
    'max_val': False,  # Use max value as center
    'cut_landscape': False,  # Take only valid result of the landscape (without padding, rolling etc.)
    'circle_cut': True,  # Set values of filter outside the circle to zero
    'steepness': [.05, .15, .25, .35, .45],  # Each value represents steepness of the gaussian filter
    'start': 0,  # From what radius to save the radial data
    'end': 35,  # Up what radius to save the radial data
    'moving_avg': 1,  # Use moving averaging to smooth the radial data (relevant if larger than 1)
    'batch_size': 2048,
    'save_dir': 'debug_testing',
    'new_centers': False,
    'outlier_removal': True,
    'save_plots': True,
}

if filtering_params['dataset_name'] == '10028':
    filtering_params['extraction_size'] = 360
    filtering_params['pixel_size'] = 1.34
    filtering_params['data_dir'] = f'{filtering_params["dataset_name"]}/'
    # filtering_params['results_dir'] = 'sanity_check5'
elif filtering_params['dataset_name'] == '10017':
    filtering_params['particle_size'] = 150
    filtering_params['pixel_size'] = 1.77
    filtering_params['data_dir'] = f'{filtering_params["dataset_name"]}/'
    # filtering_params['results_dir'] = 'sanity_clust3'
# elif filtering_params['dataset_name'] == '10005':
#     filtering_params['particle_size'] = 200
#     filtering_params['pixel_size'] = 1.2156
#     filtering_params['results_dir'] = 'sanity_clust'


mrc_files = glob.glob(filtering_params['data_dir'] + '*.mrc')
auto_pick = [ctf for ctf in glob.glob(filtering_params['data_dir'] + '*.star') if '_autopick' in ctf]
ctf_file = filtering_params['data_dir'] + 'micrographs_ctf.star'
mrc_files.sort()
auto_pick.sort()
files_list = [(mrc_files[i], auto_pick[i]) for i in range(len(mrc_files))]
if filtering_params['new_centers']:
    run_center_update(mrc_files_list=files_list, ctf_file=ctf_file, filtering_params=filtering_params)
    auto_pick = [ctf for ctf in
                 glob.glob(f'{filtering_params["save_dir"]}/{filtering_params["dataset_name"]}/*.star') if
                 '_autopick' in ctf]
    auto_pick.sort()
    files_list = [(mrc_files[i], auto_pick[i]) for i in range(len(mrc_files))]
    logger.info(
        f'New centers were extracted and saved successfully in {filtering_params["save_dir"]}/{filtering_params["dataset_name"]}')
if filtering_params['outlier_removal']:
    run_coord_update(mrc_files_list=files_list, ctf_file=ctf_file, filtering_params=filtering_params)
    logger.info(f'Outliers were removed')

