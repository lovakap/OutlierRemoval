import os.path
import time
import glob
import mrcfile
import logging

from image_filtering import ImageFilter, CLASTER_TYPES
from full_flow_utils import run_full_flow
from aspire.storage import StarFile

from coord_update import run_coord_update
import pycuda
import pycuda.driver as cuda
cuda.init()

logger = logging.getLogger(__name__)

import pandas as pd
COORD_COLS = ['_rlnCoordinateX', '_rlnCoordinateY']

# ls_names = []
# ls_values = []
# for pt in glob.glob("/home/levk/Relion/relion/build/Extract/*"):
#     if "job" in pt:
#         continue
#     ls_values.append(find_common_particles(pt + '/particles.star'))
#     ls_names.append(os.path.basename(pt))

def find_common_particles(path1, path2="/home/levk/Relion/relion/build/Select/job077/particles.star"):
    star_file_1 = StarFile(path1)
    star_file_2 = StarFile(path2)
    df1 = star_file_1.get_block_by_index(1).drop_duplicates(subset=COORD_COLS)
    df2 = star_file_2.get_block_by_index(1).drop_duplicates(subset=COORD_COLS)
    merged = pd.merge(df1[COORD_COLS], df2, on=COORD_COLS, how='inner')
    # print(f"common {merged.shape[0] / df2.shape[0]}")
    # return merged
    return merged.shape[0] / df2.shape[0], merged.shape[0] / df1.shape[0], merged.shape[0], df1.shape[0]

def update_pick_loc(star_file, orig, updated):
    df = star_file.get_block_by_index(0)
    df['_rlnMicrographCoordinates'] = df['_rlnMicrographCoordinates'].str.replace(orig, updated)
    return df

def update_star(path, old, new):
    star_f = StarFile(path)
    star_f.blocks[list(star_f.blocks.keys())[0]] = update_pick_loc(star_f, old, new)
    star_f.write(path)

ls_names = []
ls_per_orig = []
ls_per_synt = []
ls_common_shape = []
ls_synt_shape = []
# for pt in glob.glob("/home/levk/Relion/relion/build/Extract/filt_25_var/*"):
#     if "job" in pt:
#         continue
#     pers = find_common_particles(pt + '/particles.star')
#     ls_per_orig.append(pers[0])
#     ls_per_synt.append(pers[1])
#     ls_common_shape.append(pers[2])
#     ls_synt_shape.append(pers[3])
#     ls_names.append(os.path.basename(pt))
#
# df = pd.DataFrame({'filter': ls_names, 'per_of_orig': ls_per_orig, 'per_of_synt': ls_per_synt, 'common_size': ls_common_shape, 'synt_size': ls_synt_shape})
# df.to_csv('comparison/25_var.csv', index=False)
general_params = {
    'run_filtering': True,  # Use outlier filtering
    'interactive': False,  # Draw blocking interactive plots?
    'do_cov2d': False,  # Use CWF coefficients
    'use_shiny': False,  # Use Shiny dataset
    'results_dir': 'new_trial',  # Path to save results
}

reconstruction_params = {
    # 'n_imgs': 10240,  # Set to None for all images in starfile, can set smaller for tests.
    'n_imgs': 30720,  # Set to None for all images in starfile, can set smaller for tests.
    'img_size': 33,  # Downsample the images/reconstruction to a desired resolution
    'n_classes': 100,  # How many class averages to compute.
    'n_neighbors': 50,  # How many neighbors to stack
    'pixel_size': 1.34,
}

filtering_params = {
    'top_n': 3,  # How many points to use for centering
    'filter_ratio': .25,  # Filter size ratio to image
    'particle_size': reconstruction_params['img_size'],
    'max_val': False,  # Use max value as center
    'cut_landscape': True,  # Take only valid result of the landscape (without padding, rolling etc.)
    'circle_cut': True,  # Set values of filter outside the circle to zero
    'steepness': [.05, .15, .25, .35, .45],  # Each value represents steepness of the gaussian filter
    'start': 0,  # From what radius to save the radial data
    'end': 35,  # Up what radius to save the radial data
    'moving_avg': 1,  # Use moving averaging to smooth the radial data (relevant if larger than 1)
    'batch_size': 2048,
    'coord_dir': '10028/raw_pick/',
    'save_dir': general_params['results_dir'],
    'name': "Original"
    # 'name': "Good"
    # 'name': "Bad"
}
# cluster_types = [1]
# clust_operators = ['&']
# cluster_types = [1, 2, 3]
cluster_types = [1, 2, 3, 4, 5, 6]
clust_operators = ['or']
# clust_operators = ['or', 'and']

micrograph_path = "10028/"
mrc_files = glob.glob(micrograph_path + 'micrographs/*.mrc')
if filtering_params['name'] == "Original":
    auto_pick = [ctf for ctf in glob.glob(micrograph_path + 'raw_pick/*.star') if '_autopick' in ctf]
elif filtering_params['name'] == "Good":
    auto_pick = [ctf for ctf in glob.glob(micrograph_path + 'good_pick/*.star') if '_autopick' in ctf]
elif filtering_params['name'] == "Bad":
    auto_pick = [ctf for ctf in glob.glob(micrograph_path + 'anti_pick/*.star') if '_autopick' in ctf]

# auto_pick = [ctf for ctf in glob.glob(micrograph_path + 'pick_and_class/*.star') if '_autopick' in ctf]
# auto_pick = [ctf for ctf in glob.glob(micrograph_path + 'raw_pick/*.star') if '_autopick' in ctf]
auto_pick = [ctf for ctf in glob.glob(micrograph_path + 'empty/*.star') if '_autopick' in ctf]
# auto_pick = [ctf for ctf in glob.glob('new_trial/image/10028/*.star') if '_autopick' in ctf]
ctf_list = [ctf for ctf in glob.glob(micrograph_path + 'micrographs/*.star') if '_autopick' not in ctf]
mrc_files.sort()
auto_pick.sort()
ctf_list.sort()


files_list = [(mrc_files[i], auto_pick[i]) for i in range(len(mrc_files))]
# files_list = [(mrc_files[i], auto_pick[i]) for i in range(2)]

filtering_params['cluster_types'] = cluster_types
filtering_params['cluster_operator'] = clust_operators
params = {'general_params': general_params, 'reconstruction_params': reconstruction_params,
          'filtering_params': filtering_params}
run_coord_update(mrc_files_list=files_list, ctf_list=ctf_list, **params)
