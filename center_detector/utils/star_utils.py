from aspire.storage import StarFile
import numpy as np
import pandas as pd
import os


def update_star_coord_loc(star_file, orig, updated):
    df = star_file.get_block_by_index(0)
    df['_rlnMicrographCoordinates'] = df['_rlnMicrographCoordinates'].str.replace(orig, updated)
    return df


def copy_star_coord_loc(orig_path, save_path, old, new):
    star_f = StarFile(orig_path)
    star_f.blocks[list(star_f.blocks.keys())[0]] = update_star_coord_loc(star_f, old, new)
    star_f.write(save_path)


def update_star_centers(combined, updated_coords, data_dir, save_path, mrc_size=4096, particle_size=360):
    half_p = np.ceil(particle_size / 2) + 2
    index = combined[0]
    file_name = os.path.basename(combined[1])[:-4]
    coord_file = StarFile(data_dir + file_name + '_autopick.star')
    "Remove bad particles with class -999"
    temp_df = pd.DataFrame.from_dict(coord_file.get_block_by_index(0))
    valid_x = (temp_df['_rlnCoordinateX'].astype(float) >= half_p - 2).values & (
                temp_df['_rlnCoordinateX'].astype(float) <= mrc_size - half_p + 2).values
    valid_y = (temp_df['_rlnCoordinateY'].astype(float) >= half_p - 2).values & (
                temp_df['_rlnCoordinateY'].astype(float) <= mrc_size - half_p + 2).values
    valid = valid_x & valid_y

    update_coords = updated_coords[index]
    temp_df.loc[valid, '_rlnCoordinateX'] = (temp_df[valid]['_rlnCoordinateX'].astype(float).astype(int).values + \
                                             update_coords[
                                                 '_rlnCoordinateX'].values).clip(half_p, mrc_size - half_p).astype(object)
    temp_df.loc[valid, '_rlnCoordinateY'] = (temp_df[valid]['_rlnCoordinateY'].astype(float).astype(int).values + \
                                             update_coords[
                                                 '_rlnCoordinateY'].values).clip(half_p, mrc_size - half_p).astype(object)

    coord_file.blocks[list(coord_file.blocks.keys())[0]] = temp_df.reset_index(drop=True).to_dict(orient='list')
    coord_file.write(save_path + f'/{os.path.dirname(combined[1])}/' + f'{file_name}' + '_autopick.star')


def update_picks_by_star(star_path, picks_folder, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    new_picks = StarFile(star_path).blocks['particles']
    for mrc_name in new_picks['_rlnMicrographName'].unique():
        pick_star = StarFile(picks_folder + os.path.basename(mrc_name)[:-4] + '_autopick.star')
        cols = pick_star.get_block_by_index(0).columns
        pick_star.blocks[list(pick_star.blocks.keys())[0]] = new_picks[new_picks['_rlnMicrographName'] == mrc_name][cols].reset_index(drop=True)
        pick_star.write(save_folder + os.path.basename(mrc_name)[:-4] + '_autopick.star')

# print('end')