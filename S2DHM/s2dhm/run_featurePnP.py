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

from network import network
from image_retrieval import rank_images
from network import network
from pose_prediction import predictor
from datasets import base_dataset

from pose_prediction.sparse_to_dense_featuremetric_predictor import SparseToDenseFeatureMetricPnP
from datasets import dataload_helpers as data_helpers

# this is a hack
sys.path.insert(0, os.path.abspath('../../'))


from BA.featureBA.src.model import sparse3DBA
from BA.featureBA.src.utils import sobel_filter


# Argparse
parser = argparse.ArgumentParser(
    description = 'Sparse-to-dense Hypercolumn Matching')
parser.add_argument(
    '--dataset', type=str, help = 'root path to the robotcar dataset', required=True)
parser.add_argument(
    '--result', type=str, help = 'path where you want to save the results', required=True)
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
    ref_images = ['overcast-reference/rear/1417176495903575.jpg',
                'overcast-reference/rear/1417176586163507.jpg',
                'overcast-reference/rear/1417176991015329.jpg',
                'overcast-reference/rear/1417177488763211.jpg',
                'overcast-reference/rear/1417177952697851.jpg',
                'overcast-reference/rear/1417177961605723.jpg',
                'overcast-reference/rear/1417177976512750.jpg',
                'overcast-reference/rear/1417177990238067.jpg']
    query_images = ['overcast-reference/rear/1417176509083576.jpg',
                'overcast-reference/rear/1417176586345356.jpg',
                'overcast-reference/rear/1417176991197053.jpg',
                'overcast-reference/rear/1417177489035919.jpg',
                'overcast-reference/rear/1417177952879702.jpg',
                'overcast-reference/rear/1417177961787572.jpg',
                'overcast-reference/rear/1417177976603612.jpg',
                'overcast-reference/rear/1417177990601642.jpg']


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

    # Load gin config based on dataset name
    gin.parse_config_file(
        'configs/runs/run_{}_on_{}.gin'.format('sparse_to_dense', 'robotcar'))

    net = network.ImageRetrievalModel(device = "cuda")
    s2dPnP = SparseToDenseFeatureMetricPnP(filename_to_pose, filename_to_intrinsics,
            filename_to_local_reconstruction, net)

    for k in range(len(query_images)):
        prediction, query_hypercolumn, reference_hypercolumn = s2dPnP.run(query_images[k], ref_images[k])
        
        # del filename_to_local_reconstruction
        # query_hypercolumn = interpolate(query_hypercolumn, size=(1024,1024),
        #                                 mode = 'bilinear', align_corners=True)
        # reference_hypercolumn = interpolate(reference_hypercolumn, size = (1024,1024),
        #                                     mode='bilinear', align_corners = True)
        
        print("Hypercolumn size is {}".format(query_hypercolumn.shape))

        result = dict()
        result['prediction'] = prediction
        # result['query_hypercolumn'] = query_hypercolumn.numpy()
        # result['reference_hypercolumn'] = reference_hypercolumn.numpy()

        # np.save(args.result + "query_hypercolumn", query_hypercolumn.numpy())
        # np.save(args.result + "reference_hypercolumn", reference_hypercolumn.numpy())
        


        ### For plotting
        q_img = cv2.imread(query_images[k])
        r_img = cv2.imread(ref_images[k])

        for i, p in enumerate(prediction.reference_inliers.astype(int)):
            # print(i, p)
            cv2.circle(r_img, tuple(p), 1, (128, 128, 0), 3)

        cv2.imwrite(args.result + 'ref_'+ str(k) + '_detection.png', r_img)

        for i, p in enumerate(prediction.query_inliers.astype(int)):
            # print(i, p)
            cv2.circle(q_img, tuple(p), 1, (128, 128, 0), 3)

        cv2.imwrite(args.result + 'query_'+ str(k) + '_detection.png', q_img)

        pts3D = torch.from_numpy(prediction.points_3d.reshape(-1,3))
        ref2d = torch.flip(torch.from_numpy((1/8*prediction.reference_inliers).astype(int)),(1,))
        feature_ref = torch.cat([reference_hypercolumn.squeeze(0)[:, i, j].unsqueeze(0) for i, j in zip(ref2d[:,0],
                                ref2d[:,1])]).type(torch.DoubleTensor)
        feature_map_query = query_hypercolumn.squeeze(0).type(torch.DoubleTensor)
        T_init = filename_to_pose['/'.join(ref_images[k].split('/')[-3:])][1]
        # T_init = prediction.matrix
        R_init, t_init = torch.from_numpy(T_init[:3, :3]), torch.from_numpy(T_init[:3,3])
        feature_grad_x, feature_grad_y = sobel_filter(feature_map_query)
        K = torch.from_numpy(filename_to_intrinsics[ref_images[k]][0]).type(torch.DoubleTensor)

        print("3D points shape is {}".format(pts3D.size()))
        print("reference features shape is {}".format(feature_ref.size()))
        print("query feature map is of size {}".format(feature_map_query.size()))
        print("Rotation matrix is of shape {} and translation {}".format(R_init.size(), t_init.size()))
        print("feature gradient is of shape {}".format(feature_grad_x.size()))
        # inital projection and plotting
        proj2d = torch.mm(R_init, pts3D.T).T + t_init
        proj2d = torch.mm(K, proj2d.T).T
        proj2d = proj2d/proj2d[:,-1,None]
        proj2d = torch.round(proj2d[:,:2]).type(torch.IntTensor)-1

        # inital points
        q_img = cv2.imread(query_images[k])

        for i, p in enumerate(proj2d):
            # print(i, p)
            cv2.circle(q_img, tuple(p), 1, (128, 128, 0), 3)

        cv2.imwrite(args.result + 'query_' + str(k) + '_initialization.png', q_img)

        model = sparse3DBA(n_iters = 100, lambda_ = 0.1, verbose=True)
        R, t = model(pts3D, feature_ref, feature_map_query, feature_grad_x, feature_grad_y, K, 1024, 1024,R_init, t_init)

        result['R'] = R.numpy()
        result['t'] = t.numpy()
        pickle.dump(result, open(args.result +str(k) + "_prediction.p", 'wb'))

        # final projection and plotting
        q_img = cv2.imread(query_images[k])
        proj2d = torch.mm(R, pts3D.T).T + t
        proj2d = torch.mm(K, proj2d.T).T
        proj2d = proj2d/proj2d[:,-1,None]
        proj2d = torch.round(proj2d[:,:2]).type(torch.IntTensor)-1

        for i, p in enumerate(proj2d):
            # print(i, p)
            cv2.circle(q_img, tuple(p), 1, (128, 128, 0), 3)

        cv2.imwrite(args.result + 'query_' + str(k) + '_final.png', q_img)

    del filename_to_pose, filename_to_intrinsics, query_hypercolumn, reference_hypercolumn

    
