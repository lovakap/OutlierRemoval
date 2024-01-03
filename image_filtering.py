from math import ceil
from sklearn.cluster import DBSCAN
from multiprocessing import context
from pathos.pools import ProcessPool
from functools import partial
from collections import OrderedDict
from center_detector.utils.convolution_utils import apply_filter
from center_detector.utils.plot_utils import save_radial_info, save_plot_with_counts
from center_detector.utils.general_utils import get_n_points
from center_detector.utils.star_utils import *
import logging
import glob

MIN_RANGE = 1e10
MAX_RANGE = 1e-10
N_BINS = 100

logger = logging.getLogger(__name__)


class ImageFilter:
    def __init__(self, train_args):
        self.train_args = train_args
        self.train_args['filter_size'] = int(self.train_args['filter_ratio'] * self.train_args['patch_size'])
        self.train_args["f_size_threshold"] = int(self.train_args.get("filter_size") / 2)
        self.train_args["add_value"] = self.train_args["f_size_threshold"] if self.train_args["cut_landscape"] else 0
        self.train_args['stack_size'] = 24
        context._force_start_method('spawn')
        if self.train_args['save_plots']:
            os.makedirs(self.train_args['save_dir'] + '/plot_res/', exist_ok=True)

    def update_coord(self, src):
        assert self.train_args["data_dir"] is not None and self.train_args[
            "save_dir"] is not None, "Please define data_dir and save_dir"
        total_res = []
        progress_count = 0
        for i in range(0, src.get_metadata('__mrc_index').max() + 1, self.train_args["stack_size"]):
            diff = src.get_metadata('__mrc_index').max() + 1 - (i + self.train_args["stack_size"])
            stack_size = self.train_args["stack_size"] if diff >= 0 else self.train_args["stack_size"] + diff
            with ProcessPool(int(len(os.sched_getaffinity(0))/2)) as pool:
                results = pool.map(partial(extract_and_cluster, args=self.train_args),
                                   [(src.images[np.argwhere(src.get_metadata('__mrc_index') == i + j).squeeze()]._data,
                                     i + j) for j in range(stack_size)])
            total_res.extend(results)
            progress_count += np.sum([sum(r) for r in results])

        save_path = self.train_args["save_dir"] + '/' + self.train_args['dataset_name'] + '_filtered'
        os.makedirs(save_path + f'/{self.train_args["dataset_name"]}/', exist_ok=True)
        mrc_shape = src.mrc_shapes[0][0]

        for i in range(src.get_metadata('__mrc_index').max() + 1):
            f_name = src.get_metadata('__mrc_filepath')[src.get_metadata('__mrc_index') == i][0]
            coord_file = StarFile(self.train_args["data_dir"] + f'{os.path.basename(f_name)[:-4]}' + '_autopick.star')
            temp_df = pd.DataFrame.from_dict(coord_file.get_block_by_index(0))
            for j in range(np.ceil(src.particle_size * 0.6).astype(int)):
                half_p = np.ceil(src.particle_size * 0.6) - j
                valid_x = (temp_df['_rlnCoordinateX'].astype(float) >= half_p).values & (
                        temp_df['_rlnCoordinateX'].astype(float) <= mrc_shape - half_p).values
                valid_y = (temp_df['_rlnCoordinateY'].astype(float) >= half_p).values & (
                        temp_df['_rlnCoordinateY'].astype(float) <= mrc_shape - half_p).values
                valid = (valid_x & valid_y)
                if sum(valid) == total_res[i].shape[0]:
                    break

            coord_file.blocks[list(coord_file.blocks.keys())[0]] = temp_df[valid][
                total_res[i]].reset_index(drop=True).to_dict(orient='list')
            coord_file.write(save_path + f'/{self.train_args["dataset_name"]}/' + f'{os.path.basename(f_name)[:-4]}' + '_autopick.star')
        star_file = StarFile()
        mrc_names = np.unique(src.get_metadata('__mrc_filepath')).tolist()
        auto_pick_names = [
            f'AutoPick/{self.train_args["dataset_name"]}_filtered/{self.train_args["dataset_name"]}/{os.path.basename(i)[:-4]}_autopick.star'
            for i in mrc_names]
        mrc_names.sort()
        auto_pick_names.sort()
        o_dict = OrderedDict([('coordinate_files', {'_rlnMicrographName': mrc_names,
                                                    '_rlnMicrographCoordinates': auto_pick_names})])
        star_file.blocks = o_dict
        star_file.write(save_path + "/autopick.star")
        logger.info(f'Removed {len(src) - progress_count} outliers')
        return progress_count

    def update_coord_centers(self, src):
        assert self.train_args["data_dir"] is not None and self.train_args[
            "save_dir"] is not None, "Please define data_dir and save_dir"
        total_res = []
        for i in range(0, src.get_metadata('__mrc_index').max() + 1, self.train_args["stack_size"]):
            diff = src.get_metadata('__mrc_index').max() + 1 - (i + self.train_args["stack_size"])
            stack_size = self.train_args["stack_size"] if diff >= 0 else self.train_args["stack_size"] + diff
            with ProcessPool(int(len(os.sched_getaffinity(0))/2)) as pool:
                results = pool.map(partial(extract_new_centers, args=self.train_args),
                                   [src.images[np.argwhere(src.get_metadata('__mrc_index') == i + j).squeeze()]._data
                                      for j in range(stack_size)])
            total_res.extend(results)

        os.makedirs(self.train_args["save_dir"] + f'/{self.train_args["dataset_name"]}/', exist_ok=True)
        mrc_shape = src.mrc_shapes[0][0]
        particle_size = src.particle_size
        for i in range(0, src.get_metadata('__mrc_index').max() + 1, self.train_args["stack_size"]):
            diff = src.get_metadata('__mrc_index').max() + 1 - (i + self.train_args["stack_size"])
            stack_size = self.train_args["stack_size"] if diff >= 0 else self.train_args["stack_size"] + diff
            with ProcessPool(int(len(os.sched_getaffinity(0))/2)) as pool:
                pool.map(partial(update_star_centers, updated_coords=total_res, data_dir=self.train_args["data_dir"],
                                 save_path=self.train_args["save_dir"], mrc_size=mrc_shape, particle_size=particle_size),
                         [(i + j, src.get_metadata('__mrc_filepath')[src.get_metadata('__mrc_index') == i + j][0]) for j in
                          range(stack_size)])


def extract_and_cluster(items, args):
    images, index = items
    res_info = extract_info(images, args)
    non_centric = np.ones(len(res_info["radial_info"])).astype(bool)
    non_centric[res_info["non_centric"]] = False
    results = non_centric
    if args.get('save_plots'):
        if os.path.exists(f"{args.get('save_dir')}/results_table.csv"):
            df = pd.read_csv(f"{args.get('save_dir')}/results_table.csv")
            path = f"{args.get('save_dir')}/plot_res/experiment_{df.shape[0] + 1}/"
        else:
            path = f"{args.get('save_dir')}/plot_res/experiment_1/"
        save_plot_with_counts(results, res_info.get('radial_info'), index, path)

    return results


def extract_new_centers(images, args):
    images = (images - images.mean())
    new_centers_coord_x = []
    new_centers_coord_y = []
    center = ceil(images.shape[1] / 2)
    for ind, sample in enumerate(images):
        filtered_sample = apply_filter(image=sample, filter_size=args["filter_size"], circle_cut=args["circle_cut"],
                                       steepness=args["steepness"])

        sample_points = get_n_points(filtered_sample, n=args["top_n"], max_values=args["max_val"],
                                     add_value=args["add_value"])
        if len(sample_points[0]) > 1:
            dbscan_model = DBSCAN(eps=args['particle_radius'] / 2, min_samples=round(len(sample_points[0]) / 3))
            clustering_res = dbscan_model.fit_predict(np.vstack(sample_points).T)
            unique, counts = np.unique(clustering_res, return_counts=True)
            if len(unique) == 1 and unique[0] == -1:
                sample_points = (sample_points[0][:1], sample_points[1][:1])
            else:
                centers = []
                for un in unique:
                    if un == -1:
                        continue
                    p = (np.array([sample_points[0][clustering_res == un].mean().round().astype(int)]),
                         np.array([sample_points[1][clustering_res == un].mean().round().astype(int)]))
                    centers.append(p)
                if len(centers) == 1:
                    sample_points = centers[0]
                else:
                    majority_class = unique[unique != -1][counts[unique != -1].argmax()]
                    sample_points = (
                        np.array([sample_points[0][clustering_res == majority_class].mean().round().astype(int)]),
                        np.array([sample_points[1][clustering_res == majority_class].mean().round().astype(int)]))

        new_centers_coord_y.append(sample_points[0][0] - center)
        new_centers_coord_x.append(sample_points[1][0] - center)

    return pd.DataFrame({'_rlnCoordinateX': new_centers_coord_x, '_rlnCoordinateY': new_centers_coord_y})


def extract_info(images, args):
    filtered_images = []
    radial_info = []
    non_centric = []
    outlier_range = int(args['particle_radius']/2) - 1
    images = (images - images.mean())
    for ind, sample in enumerate(images):
        sample = sample / (sample.max())
        filtered_sample = apply_filter(image=sample, filter_size=args["filter_size"], circle_cut=args["circle_cut"],
                                       steepness=args["steepness"])
        if args["cut_landscape"]:
            filtered_sample = filtered_sample[args["f_size_threshold"]:-args["f_size_threshold"],
                              args["f_size_threshold"]:-args["f_size_threshold"]]

        sample_points = get_n_points(filtered_sample, n=args["top_n"], max_values=args["max_val"],
                                     add_value=args["add_value"])
        centric = True
        if len(sample_points[0]) > 1:
            dbscan_model = DBSCAN(eps=args['particle_radius'] / 2, min_samples=round(len(sample_points[0]) / 3))
            clustering_res = dbscan_model.fit_predict(np.vstack(sample_points).T)
            unique, counts = np.unique(clustering_res, return_counts=True)
            if (len(unique) == 1 and unique[0] == -1) or counts[unique == -1] > round(len(sample_points[0]) / 3):
                sample_points = (sample_points[0][:1], sample_points[1][:1])
                centric = False
            else:
                centers = []
                for un in unique:
                    if un == -1:
                        if (sample_points[0][clustering_res == -1] <= outlier_range).all() | (
                                sample_points[0][clustering_res == -1] >= sample.shape[-1] - outlier_range).all() | (
                                sample_points[1][clustering_res == -1] <= outlier_range).all() | (
                                sample_points[1][clustering_res == -1] >= sample.shape[-1] - outlier_range).all():
                            centric = False
                        continue
                    p = (np.array([sample_points[0][clustering_res == un].mean().round().astype(int)]),
                         np.array([sample_points[1][clustering_res == un].mean().round().astype(int)]))
                    centers.append(p)
                    if (p[0][0] <= outlier_range) | (p[0][0] >= sample.shape[-1] - outlier_range) | (
                            p[1][0] <= outlier_range) | (p[1][0] >= sample.shape[-1] - outlier_range):
                        centric = False
                if len(centers) == 1:
                    sample_points = centers[0]
                else:
                    majority_class = unique[unique != -1][counts[unique != -1].argmax()]
                    sample_points = (
                        np.array([sample_points[0][clustering_res == majority_class].mean().round().astype(int)]),
                        np.array([sample_points[1][clustering_res == majority_class].mean().round().astype(int)]))
        else:
            sample_points = (sample_points[0][:1], sample_points[1][:1])

        if not centric:
            non_centric.append(ind)

        rad_info = save_radial_info(patches=[sample], labels=['Sample'],
                                    filter_size=args["filter_size"],
                                    points=[sample_points], path='', plot=False, return_vals=True)

        r_mean = np.convolve(rad_info['Sample']['radial_mean'][args["start"]:args["end"]],
                        np.ones(args["moving_avg"]) / args["moving_avg"], 'same')
        r_mean = r_mean/r_mean.max()
        radial_info.append(r_mean)
        filtered_images.append(filtered_sample.flatten())

    return {"filtered_images": filtered_images, "radial_info": radial_info,  "non_centric": non_centric}

