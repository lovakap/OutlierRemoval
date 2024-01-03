import os.path
import time

import logging
import pandas as pd
import numpy as np
from aspire.source import CentersCoordinateSource

from image_filtering import ImageFilter
from typing import List, Tuple

logger = logging.getLogger(__name__)


def run_coord_update(mrc_files_list: List[Tuple[str, str]], ctf_file: str, filtering_params):
    start_test_time = time.time()

    src = CentersCoordinateSource(mrc_files_list, particle_size=filtering_params['extraction_size'])
    src.import_relion_ctf(ctf_file)

    os.makedirs(filtering_params['save_dir'], exist_ok=True)
    # Downsample the images
    logger.info(f"Set the resolution to {filtering_params['patch_size']} X {filtering_params['patch_size']}")
    # Use phase_flip to attempt correcting for CTF.
    src = src.phase_flip()
    src = src.downsample(filtering_params['patch_size'])

    logger.info("Perform phase flip to input images.")

    preprocess_time = (time.time() - start_test_time) / 60
    start_filt_time = time.time()
    img_filter = ImageFilter(filtering_params)
    prog_count = img_filter.update_coord(src)
    filt_time = time.time() - start_filt_time

    logger.info(f'Filtering time {filt_time}')
    logger.info(f'Total time : {time.time() - start_test_time}')
    COLS = ['experiment', 'filt_time', 'preprocess_time', 'total_images', 'post_filtering']
    if os.path.exists(filtering_params['save_dir'] + '/results_table.csv'):
        results_csv = pd.read_csv(filtering_params['save_dir'] + '/results_table.csv')
    else:
        results_csv = pd.DataFrame(
            columns=COLS)
    results_csv = pd.concat([results_csv, pd.DataFrame(data=[
        [results_csv.shape[0] + 1, np.round(filt_time / 60, 5), np.round(preprocess_time, 5), len(src),
         prog_count]], columns=COLS)])
    results_csv.to_csv(filtering_params['save_dir'] + '/results_table.csv', index=False)
