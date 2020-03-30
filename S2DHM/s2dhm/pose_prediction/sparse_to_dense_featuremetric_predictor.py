"""Sparse-To-Dense Predictor Class.
"""
import gin
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from network.network import ImageRetrievalModel
from pose_prediction import predictor
from pose_prediction import solve_pnp
from pose_prediction import keypoint_association
from pose_prediction import exhaustive_search
from visualization import plot_correspondences

class SparseToDenseFeatureMetricPnP:
    """Sparse-to-dense Feature-Metric PnP Predictor 
    TODO:
    Make it inherit from Base Predictor
    """
    def __init__(self, filename_to_pose, filename_to_intrinsics, filename_to_local_reconstruction,
                network):
        """Initialize class attributes.

        Args:
            
        """
        self._filename_to_pose = \
            filename_to_pose
        self._filename_to_intrinsics = \
            filename_to_intrinsics
        self._filename_to_local_reconstruction = \
            filename_to_local_reconstruction
        self._network = network

    def _compute_sparse_reference_hypercolumn(self, reference_image,
                                              local_reconstruction):
        """Compute hypercolumns at every visible 3D point reprojection."""
        reference_dense_hypercolumn, image_size = \
            self._network.compute_hypercolumn(
                [reference_image], to_cpu=False, resize=True)
        dense_keypoints, cell_size = keypoint_association.generate_dense_keypoints(
            (reference_dense_hypercolumn.shape[2:]),
            Image.open(reference_image).size[::-1], to_numpy=True)
        dense_keypoints = torch.from_numpy(dense_keypoints)
        reference_sparse_hypercolumns = \
            keypoint_association.fast_sparse_keypoint_descriptor(
                [local_reconstruction.points_2D.T],
                dense_keypoints, reference_dense_hypercolumn)[0]
        return reference_sparse_hypercolumns, cell_size, reference_dense_hypercolumn

    def run(self, query_image, reference_image):
        """Run the sparse-to-dense pose predictor."""

        print('>> Generating pose predictions using sparse-to-dense matching...')
        output = []

        query_dense_hypercolumn, _ = self._network.compute_hypercolumn(
            [query_image], to_cpu=False, resize=True)
        channels, width, height = query_dense_hypercolumn.shape[1:]
        query_dense_hypercolumn_2 = query_dense_hypercolumn.squeeze().view(
            (channels, -1))
        predictions = []

        local_reconstruction = \
            self._filename_to_local_reconstruction[reference_image]
        reference_sparse_hypercolumns, cell_size, reference_dense_hypercolumn = \
            self._compute_sparse_reference_hypercolumn(
                reference_image, local_reconstruction)

        # Perform exhaustive search
        matches_2D, mask = exhaustive_search.exhaustive_search(
            query_dense_hypercolumn_2,
            reference_sparse_hypercolumns,
            Image.open(reference_image).size[::-1],
            [width, height],
            cell_size)

        # Solve PnP
        points_2D = np.reshape(
            matches_2D.cpu().numpy()[mask], (-1, 1, 2))
        points_3D = np.reshape(
            local_reconstruction.points_3D[mask], (-1, 1, 3))
        distortion_coefficients = \
            local_reconstruction.distortion_coefficients
        intrinsics = local_reconstruction.intrinsics
        prediction = solve_pnp.solve_pnp(
            points_2D=points_2D,
            points_3D=points_3D,
            intrinsics=intrinsics,
            distortion_coefficients=distortion_coefficients,
            reference_filename=reference_image,
            reference_2D_points=local_reconstruction.points_2D[mask],
            reference_keypoints=None)

        # If PnP failed, fall back to nearest-neighbor prediction
        if not prediction.success:
            prediction = self._nearest_neighbor_prediction(
                reference_image)
            if prediction:
                return prediction, query_dense_hypercolumn, reference_dense_hypercolumn
        else:
            return prediction, query_dense_hypercolumn, reference_dense_hypercolumn

    @property
    def dataset(self):
        return self._dataset
