import glob
import logging
import argparse

from cluster_images import run_coord_update
from center_images import run_center_update

logger = logging.getLogger('aspire.storage')
logger.propagate = False
COORD_COLS = ['_rlnCoordinateX', '_rlnCoordinateY']

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process filtering parameters for cryo-EM data.")

    # Filtering params
    parser.add_argument('--dataset_name', type=str, default='10028', help="Name of the dataset")
    parser.add_argument('--extraction_size', type=int, default=360, help="Patch size to crop from the micrograph")
    parser.add_argument('--patch_size', type=int, default=33, help="Down-sample the images/reconstruction to a desired resolution")
    parser.add_argument('--pixel_size', type=float, default=1.34, help="Pixel size of the micrograph")
    parser.add_argument('--data_dir', type=str, default='10028/', help="Directory with mrc, coordinates, and CTF files")
    parser.add_argument('--top_n', type=int, default=10, help="How many points to use for centering")
    parser.add_argument('--filter_ratio', type=float, default=0.3, help="Filter size ratio to image")
    parser.add_argument('--particle_radius', type=int, default=12, help="Approximate radius of a particle")
    parser.add_argument('--max_val', action='store_true', help="Use max value as center")
    parser.add_argument('--cut_landscape', action='store_true', help="Take only valid result of the landscape")
    parser.add_argument('--circle_cut', action='store_true', help="Set values of filter outside the circle to zero")
    parser.add_argument('--steepness', nargs='+', type=float, default=[0.05, 0.15, 0.25, 0.35, 0.45], help="Each value represents steepness of the gaussian filter")
    parser.add_argument('--start', type=int, default=0, help="From what radius to save the radial data")
    parser.add_argument('--end', type=int, default=35, help="Up what radius to save the radial data")
    parser.add_argument('--moving_avg', type=int, default=1, help="Use moving averaging to smooth the radial data")
    parser.add_argument('--batch_size', type=int, default=2048, help="Batch size")
    parser.add_argument('--save_dir', type=str, default='results', help="Directory to save the results")
    parser.add_argument('--new_centers', action='store_true', help="Generate new centers for particles")
    parser.add_argument('--outlier_removal', action='store_true', help="Enable outlier removal")
    parser.add_argument('--save_plots', action='store_true', help="Save plots of the results")

    args = parser.parse_args()
    return args


def execute_experiment(args):
    filtering_params = vars(args)
    if filtering_params['dataset_name'] == '10028':
        filtering_params['extraction_size'] = 280
        filtering_params['pixel_size'] = 1.34
        filtering_params['data_dir'] = f'{filtering_params["dataset_name"]}/'
    elif filtering_params['dataset_name'] == '10017':
        filtering_params['extraction_size'] = 100
        filtering_params['pixel_size'] = 1.77
        filtering_params['data_dir'] = f'{filtering_params["dataset_name"]}/'

    mrc_files = glob.glob(filtering_params['data_dir'] + '*.mrc')
    ctf_file = filtering_params['data_dir'] + 'micrographs_ctf.star'
    mrc_files.sort()
    auto_pick = [mrc[:-4] + '_autopick.star' for mrc in mrc_files]
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

