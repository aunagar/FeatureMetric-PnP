import os, sys
import numpy, torch
import gin
from pose_prediction import matrix_utils

# this is a hack
sys.path.insert(0, os.path.abspath('../../'))


from BA.featureBA.src.model import sparse3DBA
from BA.featureBA.src.utils import sobel_filter

@gin.configurable
def optimize(query_hypercolumns, net, prediction, K, image_shape):

    reference_hypercolumns = net.compute_hypercolumn( [prediction.reference_filename], to_cpu=False, resize=True )
    relative_shape = np.array([(image_shape[0]/reference_hypercolumns.shape[2]).astype(float), (image_shape[1]/reference_hypercolumns.shape[3]).astype(float)])

    pts3D = torch.from_numpy( prediction.points_3d.reshape(-1,3) )
    ref2d = torch.dot( relative_shape, prediction.reference_inliers.T ).T
    ref2d = torch.flip( torch.from_numpy(ref2d.astype(int)), (1,) )

    feature_ref = torch.cat([reference_hypercolumn.squeeze(0)[:, i, j].unsqueeze(0) for i, j in zip(ref2d[:,0],ref2d[:,1])]).type(torch.DoubleTensor)
    feature_map_query = query_hypercolumns.squeeze(0).type(torch.DoubleTensor)
    # T_init = filename_to_pose['/'.join(ref_images[0].split('/')[-3:])][1]
    T_init = prediction.matrix
    R_init, t_init = torch.from_numpy(T_init[:3, :3]), torch.from_numpy(T_init[:3,3])
    feature_grad_x, feature_grad_y = sobel_filter(feature_map_query)

    model = sparse3DBA(n_iters = 500, lambda_ = 0.1, verbose=True)
    R, t = model(pts3D, feature_ref, feature_map_query, feature_grad_x, feature_grad_y,
    K, image_shape[0], image_shape[1],R_init, t_init)

    T = np.eye(4)
    T[:3, :3], T[3,:3] = R.numpy(), t.numpy()

    quaterion = matrix_utils.matrix_quaternion(T)
    return T[:3, :3], quaterion
