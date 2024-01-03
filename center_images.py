import os.path
import time

import logging

from aspire.source import CentersCoordinateSource

from image_filtering import ImageFilter
from typing import List, Tuple


logger = logging.getLogger(__name__)


def run_center_update(mrc_files_list: List[Tuple[str, str]], ctf_file: str, filtering_params):
    start_test_time = time.time()

    src = CentersCoordinateSource(mrc_files_list, particle_size=filtering_params['extraction_size'])
    src.import_relion_ctf(ctf_file)

    os.makedirs(filtering_params['save_dir'], exist_ok=True)
    # Downsample the images
    logger.info(f"Set the resolution to {filtering_params['patch_size']} X {filtering_params['patch_size']}")
    # Use phase_flip to attempt correcting for CTF.
    src.phase_flip()
    src = src.downsample(filtering_params['patch_size'])

    logger.info("Perform phase flip to input images.")
    start_filt_time = time.time()
    image_filter = ImageFilter(filtering_params)
    image_filter.update_coord_centers(src)
    filt_time = time.time() - start_filt_time
    logger.info(f'Centering time {filt_time}')
    logger.info(f'Total time : {time.time() - start_test_time}')
