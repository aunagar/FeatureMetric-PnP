import os
import numpy as np
import gin
import argparse
from torch.nn.functional import interpolate
from PIL import Image
import pickle

from network import network
from image_retrieval import rank_images
from network import network
from pose_prediction import predictor
from datasets import base_dataset

from pose_prediction.sparse_to_dense_featuremetric_predictor import SparseToDenseFeatureMetricPnP
from datasets import dataload_helpers as data_helpers

# Argparse
parser = argparse.ArgumentParser(
    description = 'Sparse-to-dense Hypercolumn Matching')
parser.add_argument(
    '--dataset', type=str, help = 'root path to the robotcar dataset', required=True)
parser.add_argument(
    '--ref_image', type = str, help = 'reference image path (relative to Image folder)', required = False)
parser.add_argument(
    '--query_image', type = str, help = 'query image path (relative to Image folder)', required = False)


def load_data(triangulation_filepath, nvm_filepath, filenames):
    filename_to_pose = data_helpers.from_nvm(nvm_filepath, filenames)
    filename_to_local_reconstruction = data_helpers.load_triangulation_data(triangulation_filepath, filenames)
    filename_to_intrinsics = data_helpers.load_intrinsics(filenames)

    return filename_to_pose, filename_to_local_reconstruction, filename_to_intrinsics

##### You can ignore this part #####
@gin.configurable
def get_dataset_loader(dataset_loader_cls):
    return dataset_loader_cls

@gin.configurable
def get_pose_predictor(pose_predictor_cls: predictor.PosePredictor,
                       dataset: base_dataset.BaseDataset,
                       network: network.ImageRetrievalModel,
                       ranks: np.ndarray,
                       log_images: bool):
    return pose_predictor_cls(dataset=dataset,
                              network=network,
                              ranks=ranks,
                              log_images=log_images)

if __name__ == '__main__':
    args = parser.parse_args()
    DATA_PATH = args.dataset

    # put triangulation file in the same folder as robotcar data
    triangulation_file = DATA_PATH + 'robotcar_triangulation.npz'
    nvm_filepath = DATA_PATH + '3D-models/all-merged/all.nvm'
    image_root = DATA_PATH + 'images/'

    # if no file is provided in args, we will use our own!
    ref_image = 'overcast-reference/rear/1417176458999821.jpg'
    query_image = 'overcast-reference/rear/1417176511810439.jpg'

    if args.ref_image:
        ref_images = [image_root + args.ref_image]
    else:
        print("using default image : {}".format(ref_image))
        ref_images = [image_root + ref_image]

    if args.query_image:
        query_images = [image_root + args.query_image]
    else:
        print("using default image : {}".format(query_image))
        query_images = [image_root + query_image]

    filenames = ref_images + query_images

    filename_to_pose, filename_to_local_reconstruction,\
    filename_to_intrinsics = load_data(triangulation_file, nvm_filepath, filenames)

    # Load gin config based on dataset name
    gin.parse_config_file(
        'configs/runs/run_{}_on_{}.gin'.format('sparse_to_dense', 'robotcar'))

    net = network.ImageRetrievalModel(device = "cpu")
    s2dPnP = SparseToDenseFeatureMetricPnP(filename_to_pose, filename_to_intrinsics,
            filename_to_local_reconstruction, net)
    prediction, query_hypercolumn, reference_hypercolumn = s2dPnP.run(query_images[0], ref_images[0])

    result = dict()
    result['prediction'] = prediction
    result['query_hypercolumn'] = query_hypercolumn
    result['reference_hypercolumn'] = reference_hypercolumn

    pickle.dump(result, open("results/prediction.p", 'wb'))