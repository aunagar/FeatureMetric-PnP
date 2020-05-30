import os
import sys
import gin
import torch
import argparse
import logging
import numpy as np


sys.path.append('s2dhm/') #Should be autodetected later in __init__.py file!
sys.path.append('featurePnP/')
sys.path.append('visualization/')
sys.path.append('externals/d2-net/') # if you have a d2-net directory

from image_retrieval import rank_images
from network import network
from pose_prediction import predictor
from datasets import base_dataset
from input_configs.IOgin import IOgin

# Argparse
parser = argparse.ArgumentParser(
    description = 'Sparse-to-dense Hypercolumn Matching')
parser.add_argument(
    '--input_config', type = str, help = 'path to gin config file', default = "input_configs/default_robotcar.gin", required = False)
parser.add_argument(
    '--dataset', type=str, choices=['robotcar', 'cmu'], required=False, default ="robotcar")
parser.add_argument(
    '--gpu-id', help='GPU ID, if not specified all available GPUs will be used')
parser.add_argument(
    '--mode', type=str, choices=['nearest_neighbor', 'superpoint', 'sparse_to_dense'],
    default='sparse_to_dense')
parser.add_argument('--log_images', action='store_true')
parser.add_argument('--cmu_slice', type=int, default=2)

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

def bind_cmu_parameters(cmu_slice, mode):
    """Update CMU gin parameters to match chosen slice."""
    gin.bind_parameter('ExtendedCMUDataset.cmu_slice', cmu_slice)
    if mode=='nearest_neighbor':
        gin.bind_parameter('NearestNeighborPredictor.output_filename',
            '../results/cmu/slice_{}/top_1_predictions.txt'.format(cmu_slice))
    elif mode=='superpoint':
        gin.bind_parameter('SuperPointPredictor.output_filename',
            '../results/cmu/slice_{}/superpoint_predictions.txt'.format(cmu_slice))
        gin.bind_parameter('plot_correspondences.plot_correspondences.export_folder',
            '../logs/superpoint/correspondences/cmu/slice_{}/'.format(cmu_slice))
        gin.bind_parameter('plot_correspondences.plot_detections.export_folder',
            '../logs/superpoint/detections/cmu/slice_{}/'.format(cmu_slice))
        gin.bind_parameter('plot_correspondences.plot_image_retrieval.export_folder',
            '../logs/superpoint/nearest_neighbor/cmu/slice_{}/'.format(cmu_slice))
    elif mode=='sparse_to_dense':
        # gin.bind_parameter('SparseToDensePredictor.output_filename',
        #     '../results/cmu/slice_{}/sparse_to_dense_predictions.txt'.format(cmu_slice))
        # gin.bind_parameter('SparseToDensePredictor.output_file',
        #     '../results/cmu/slice_{}/'.format(cmu_slice))
        gin.bind_parameter('plot_correspondences.plot_correspondences.export_folder',
            '../logs/sparse_to_dense/correspondences/cmu/slice_{}/'.format(cmu_slice))
        gin.bind_parameter('plot_correspondences.plot_image_retrieval.export_folder',
            '../logs/sparse_to_dense/nearest_neighbor/cmu/slice_{}/'.format(cmu_slice))

def main(args):
    # Define visible GPU devices
    print('>> Found {} GPUs to use.'.format(torch.cuda.device_count()))
    if args.gpu_id:
        print('>> Using GPU {}'.format(args.gpu_id))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    gin.parse_config_file(
        args.input_config)
    
    # Load gin config based on dataset name
    # gin.parse_config_file(
    #    'configs/runs/run_{}_on_{}.gin'.format(args.mode, args.dataset))


    # For CMU, pick a slice
    if args.dataset=='cmu':
        bind_cmu_parameters(args.cmu_slice, args.mode)
    
    io_gin = IOgin(args.input_config) # Parameters from Input file (Load correct dataset)

    # Create dataset loader
    dataset = get_dataset_loader()

    # Load retrieval model and initialize pose predictor
    net = network.ImageRetrievalModel(device=device)
    print(gin.operative_config_str())

    # Check if image retrieval rankings exist, if not compute them
    ranks = rank_images.fetch_or_compute_ranks(dataset, net)

    # Predict query images poses
    pose_predictor = get_pose_predictor(dataset=dataset,
                                        network=net,
                                        ranks=ranks,
                                        log_images=args.log_images)

    pose_predictor.save(pose_predictor.run())

    with open(io_gin.output_dir + "input_operative_str.txt", "w") as doc:
        doc.write(str(gin.operative_config_str()))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
