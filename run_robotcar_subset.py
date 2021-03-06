
import os
import sys
import numpy as np
import gin
import argparse
from torch.nn.functional import interpolate
from PIL import Image
import pickle
import torch
import cv2
import pandas as pd

sys.path.append('s2dhm/') #Should be autodetected later in __init__.py file!
sys.path.append('featurePnP/')
sys.path.append('visualization/')

from datasets import base_dataset
from datasets import dataload_helpers as data_helpers
from image_retrieval import rank_images
from network import network
from pose_prediction import predictor
from pose_prediction.sparse_to_dense_featuremetric_predictor import SparseToDenseFeatureMetricPnP
from pose_prediction.optimize_feature_pnp import feature_pnp

# for correspondances
from pose_prediction import exhaustive_search
from pose_prediction.keypoint_association import kpt_to_cv2
from helpers.utils import from_homogeneous, to_homogeneous
from visualization import plot_correspondences

from visualize_hc import visualize_hc
from input_configs.IOgin import IOgin

# Argparse
parser = argparse.ArgumentParser(
    description = 'Sparse-to-dense Hypercolumn Matching')
parser.add_argument(
    '--input_config', type = str, help = 'path to gin config file', default = "input_configs/subset/default_subset.gin", required = False)
parser.add_argument(
    "--output", type=str,help = 'what output the run should generate', choices=['all', 'video', 'visualize_hc', 'correspondences','None'],
    default='None')
parser.add_argument(
    '--ref_image', type = str, help = 'reference image path (relative to Image folder)', required = False)
parser.add_argument(
    '--query_image', type = str, help = 'query image path (relative to Image folder)', required = False)
parser.add_argument(
    '--writetrack', type=bool, help = 'Whether the track should be written', required=False, default=False)

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
    DATA_PATH = "/nfs/nas12.ethz.ch/fs1201/infk_ivc_students/252-0579-00L/ntselepidis/S2DHM_datasets/RobotCar-Seasons/"
    
    result_frame = pd.DataFrame(columns=["reference_image_origin", "query_image_origin","num_initial_matches", "num_final_matches", "initial_cost", "final_cost","track_pickle_path"])

    # put triangulation file in the same folder as robotcar data
    triangulation_file = DATA_PATH + 'data/triangulation/robotcar_triangulation.npz'
    nvm_filepath = DATA_PATH + '3D-models/all-merged/all.nvm'
    image_root = DATA_PATH + 'images/'

    gin.parse_config_file(
        args.input_config)
    
    
    io_gin = IOgin(args.input_config) # Parameters from Input file

    ref_images = ['overcast-reference/rear/1417177821443708.jpg',
                  'overcast-reference/rear/1417178379637791.jpg',
                  'overcast-reference/rear/1417177965150722.jpg',
                  'overcast-reference/rear/1417176981834751.jpg',
                  'overcast-reference/rear/1417176982016599.jpg',
                  'overcast-reference/rear/1417177964241727.jpg',
                  'overcast-reference/rear/1417176981834751.jpg',
                  'overcast-reference/rear/1417178468625282.jpg',
                  'overcast-reference/rear/1417177029555384.jpg',
                  'overcast-reference/rear/1417177069186036.jpg',
                  'overcast-reference/rear/1417176982016599.jpg',
                  'overcast-reference/rear/1417177030646106.jpg',
                  'overcast-reference/rear/1417178467988996.jpg',
                  'overcast-reference/rear/1417177029100823.jpg',
                  'overcast-reference/rear/1417176982562024.jpg',
                  'overcast-reference/rear/1417178029141715.jpg',
                  'overcast-reference/rear/1417177630834182.jpg',
                  'overcast-reference/rear/1417177088547051.jpg',
                  'overcast-reference/rear/1417177821443708.jpg',
                  'overcast-reference/rear/1417177165990619.jpg']
                  
    query_images = ['night-rain/rear/1418841346403822.jpg',
                  'night-rain/rear/1418841677987976.jpg',
                  'night-rain/rear/1418841403271814.jpg',
                  'night-rain/rear/1418840689734892.jpg',
                  'night-rain/rear/1418840689859877.jpg',
                  'night-rain/rear/1418841402646891.jpg',
                  'night-rain/rear/1418840689484924.jpg',
                  'night-rain/rear/1418841708859176.jpg',
                  'night-rain/rear/1418840728855054.jpg',
                  'night-rain/rear/1418840746477876.jpg',
                  'night-rain/rear/1418840690109846.jpg',
                  'night-rain/rear/1418840729479977.jpg',
                  'night-rain/rear/1418841707859299.jpg',
                  'night-rain/rear/1418840728730070.jpg',
                  'night-rain/rear/1418840690234831.jpg',
                  'night-rain/rear/1418841595623116.jpg',
                  'night-rain/rear/1418841295785077.jpg',
                  'night-rain/rear/1418840761601006.jpg',
                  'night-rain/rear/1418841346653791.jpg',
                  'night-rain/rear/1418840815469348.jpg']

    if args.ref_image:
        ref_images = [image_root + args.ref_image]
    else:
        print("using default images")
        ref_images = [image_root + ref_image for ref_image in ref_images]

    if args.query_image:
        query_images = [image_root + args.query_image]
    else:
        print("using default images")
        query_images = [image_root + query_image for query_image in query_images]

    

    filenames = ref_images + query_images

    filename_to_pose, filename_to_local_reconstruction,\
    filename_to_intrinsics = load_data(triangulation_file, nvm_filepath, filenames)

    track = args.output in ["all", "video"] or args.writetrack

    

    net = network.ImageRetrievalModel(device = "cuda")
    
    s2dPnP = SparseToDenseFeatureMetricPnP(filename_to_pose, filename_to_intrinsics,
            filename_to_local_reconstruction, net)

    for k in range(len(query_images)):
        prediction, query_hypercolumn, reference_hypercolumn = s2dPnP.run(query_images[k], ref_images[k])

        K = torch.from_numpy(filename_to_intrinsics[ref_images[k]][0]).type(torch.DoubleTensor)
        R, t, model = feature_pnp(query_hypercolumn, reference_hypercolumn, prediction, K, (1024, 1024))

        if args.output in ["all", "correspondences"]:
            outlier_threshold = 2
            local_reconstruction = filename_to_local_reconstruction[ref_images[k]]
            pts_3d = local_reconstruction.points_3D
            ref_2d = local_reconstruction.points_2D
            R_init = prediction.matrix[:3,:3]
            t_init = prediction.matrix[:3, 3]
            reference_sparse_hypercolumn, cell_size, _ = s2dPnP._compute_sparse_reference_hypercolumn(ref_images[k],
                                                                local_reconstruction)
            channels, width, height = query_hypercolumn.shape[1:]
            matches_2D, mask = exhaustive_search.exhaustive_search(
                query_hypercolumn.squeeze().view((channels, -1)),
                reference_sparse_hypercolumn,
                Image.open(ref_images[k]).size[::-1],
                [width, height],
                cell_size)
            matches_2D = matches_2D.cpu().numpy()

            init_query_2d = np.matmul(R_init,pts_3d.T).T + t_init
            init_query_2d = np.round(from_homogeneous(np.matmul(K.numpy(), init_query_2d.T).T)).astype(int)-1

            final_query_2d = np.matmul(R.numpy(), pts_3d.T).T + t.numpy()
            final_query_2d = np.round(from_homogeneous(np.matmul(K.numpy(), final_query_2d.T).T)).astype(int)-1
            
            init_outliers = np.sqrt(np.mean(np.square(init_query_2d - matches_2D), -1)) > outlier_threshold
            final_outlier = np.sqrt(np.mean(np.square(final_query_2d - matches_2D), -1)) > outlier_threshold
            # mask = np.sqrt(np.mean(np.square(init_query_2d - matches_2D), -1)) > 2
            plot_correspondences.plot_all_points(query_images[k], ref_images[k], kpt_to_cv2(ref_2d[mask]),
                                                kpt_to_cv2(init_query_2d[mask]),
                                                init_outliers[mask], title = 'before optimization matches',
                                                export_folder = io_gin.output_dir + "matching/",
                                                export_filename = "initial_matches_" + str(k) + ".jpg" )
            plot_correspondences.plot_all_points(query_images[k], ref_images[k], kpt_to_cv2(ref_2d[mask]),
                                                kpt_to_cv2(final_query_2d[mask]),
                                                final_outlier[mask], title = 'after optimization matches',
                                                export_folder = io_gin.output_dir + "matching/",
                                                export_filename = "final_matches_" + str(k) + ".jpg")

        if args.output in ["all", "visualize_hc"]:
            ref_idx = 0 #Change to select different point!
            ref_p = prediction.reference_inliers[ref_idx]
            cv2.circle(r_img, tuple(ref_p.astype(int)), 2, (256, 0, 0), 3)
            scale = reference_hypercolumn.shape[-1]/r_img.shape[1]
            ref_p = (scale*ref_p).astype(int)
            query_p = prediction.query_inliers[ref_idx]
            cv2.circle(q_img, tuple(query_p.astype(int)), 2, (256, 0, 0), 3)
            query_p = (scale*query_p).astype(int)
            r_hc = reference_hypercolumn[:, :, ref_p[1], ref_p[0]].cpu()
            visualize_hc(r_hc, query_hypercolumn.squeeze(0).cpu(), query_p, io_gin.output_dir + 'hc_'+str(k) + '.jpg',
                        q_img, r_img)

        if args.output in ["all", "video"]:
            frames = frames_from_track(query_image, track_dict, 50)
            save_video(frames,io_gin.output_dir+"_video_"+str(k)+".mp4") #Change

        if args.writetrack:
            track_pickle_path = io_gin.output_dir + 'query_'+ str(k) + '_track.p'
            pickle.dump(model.track_, open(track_pickle_path,"wb"))
        else:
            track_pickle_path = None

        """Write to DataFrame"""
        result_frame.loc[k] = [ref_images[k], query_images[k], prediction.num_matches, model.best_num_inliers_, model.initial_cost_.item(), model.best_cost_.item(), track_pickle_path]


    result_frame.to_csv(io_gin.output_dir + io_gin.csv_name, sep = ";", index = False)

    print(gin.operative_config_str())

    with open(io_gin.output_dir + "input_operative_str.txt", "w") as doc:
        doc.write(str(gin.operative_config_str()))

    del filename_to_pose, filename_to_intrinsics, query_hypercolumn, reference_hypercolumn

    
