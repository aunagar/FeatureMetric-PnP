"""Sparse-To-Dense Predictor Class.
"""
import gin
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pose_prediction import predictor
from pose_prediction import solve_pnp
from pose_prediction import keypoint_association
from pose_prediction import exhaustive_search
from visualization import plot_correspondences
from typing import List

from pose_prediction.optimize_feature_pnp import optimize_feature_pnp
import pandas as pd
import pickle
import time

# for d2-net (import only if avaible) -- if you want to use d2-net features
# this module should be visible (path is already added to train.py)
try:
    from s2dhm.d2net_utils.extract_dense_features import extract_dense_features
except ModuleNotFoundError:
    print("Did not find the module required for d2-net features.")
    pass

def read_npz(dirpath):
    filenames = []
    for (dp, dn, fn) in os.walk(dirpath + "cached_matches/"):
        filenames.extend(fn)
        break
    files = dict()
    for fn in filenames:
        npz = np.load(dirpath + "cached_matches/" + fn,allow_pickle = True)
        npz = {key:npz[key].item() for key in npz}
        files.update(npz)
    return files

@gin.configurable
class SparseToDensePredictor(predictor.PosePredictor):
    """Sparse-to-dense Predictor Class.
    """
    def __init__(self, top_N: int, track: bool, features: str, **kwargs): #Revert here
        """Initialize class attributes.

        Args:
            top_N: Number of nearest neighbors to consider in the
                sparse-to-dense matching.
        """
        super().__init__(**kwargs)
        self._top_N = top_N
        self._filename_to_pose = \
            self._dataset.data['reconstruction_data'].filename_to_pose
        self._filename_to_intrinsics = \
            self._dataset.data['filename_to_intrinsics']
        self._filename_to_local_reconstruction = \
            self._dataset.data['filename_to_local_reconstruction']
        self.track = track
        self.features = features # Added this variable to sparse-to-dense gin (default is 's2dhm')

    def _compute_sparse_reference_hypercolumn(self, reference_image,
                                              local_reconstruction):
        """Compute hypercolumns at every visible 3D point reprojection."""
        if self.features == 'd2-net':
            reference_dense_hypercolumn = extract_dense_features(
                [reference_image], to_cpu = False)
        else:
            reference_dense_hypercolumn, image_size = \
                self._network.compute_hypercolumn(
                    [reference_image], to_cpu=False, resize=True)

        dense_keypoints, cell_size = keypoint_association.generate_dense_keypoints(
            (reference_dense_hypercolumn.shape[2:]),
            Image.open(reference_image).size[::-1], to_numpy=True)
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        dense_keypoints = torch.from_numpy(dense_keypoints).to(device)
        reference_sparse_hypercolumns = \
            keypoint_association.fast_sparse_keypoint_descriptor(
                [local_reconstruction.points_2D.T],
                dense_keypoints, reference_dense_hypercolumn)[0]
        return reference_sparse_hypercolumns, cell_size

    def run(self, start=-1, end=-1):
        """Run the sparse-to-dense pose predictor."""

        print('>> Generating pose predictions using sparse-to-dense matching...')
        if self._only_optimization:
            try:
                file_dict = read_npz(self._output_path)
            except FileNotFoundError:
                raise("Make sure .npz data is available in output_path/cached_matches/ folder")

        output = []

        tqdm_bar = tqdm(enumerate(self._ranks.T), total=self._ranks.shape[1],
                        unit='images', leave=True)
        #print('shape')
        #print(self._ranks.shape)
        # Pandas DF
        result_frame = pd.DataFrame(columns=["reference_image_origin", "query_image_origin","num_initial_matches", "num_final_matches", "initial_cost", "final_cost","track_pickle_path"])
        cnt = -1

        if self._cache_results:
            cache_dict = dict()

        # If start and end are unset -> Run on whole slice
        if (start == -1):
            start = 0
        if (end == -1):
            end = self._ranks.shape[1]-1

        for i, rank in tqdm_bar: #For each query image
            # Compute the query dense hypercolumn
            cnt+=1

            if ( cnt < start or cnt > end):
                print('Skipping cnt = %d' % cnt)
                continue

            print("Working on cnt = %d" % cnt)
            
            query_image = self._dataset.data['query_image_names'][i] #Name
            if query_image not in self._filename_to_intrinsics:
                continue
            
            # only if self.features is d2-net (we use d2-net features)
            if self.features == 'd2-net':
                query_dense_hypercolumn = extract_dense_features(
                                [query_image], to_cpu = False, resize = True)
            else: # else s2dhm features
                query_dense_hypercolumn, _ = self._network.compute_hypercolumn(
                    [query_image], to_cpu=False, resize=True)
            channels, width, height = query_dense_hypercolumn.shape[1:]
            query_dense_hypercolumn = query_dense_hypercolumn.squeeze().view(
                (channels, -1))
            predictions = []

            for j in rank[:self._top_N]: #For n reference images
                nearest_neighbor = self._dataset.data['reference_image_names'][j]

                intrinsics = self._filename_to_local_reconstruction[nearest_neighbor].intrinsics
                if self._only_optimization == False:
                    print("runnning exhaustive matching")
                    local_reconstruction = \
                        self._filename_to_local_reconstruction[nearest_neighbor]
                    reference_sparse_hypercolumns, cell_size = \
                        self._compute_sparse_reference_hypercolumn(
                            nearest_neighbor, local_reconstruction)

                    # Perform exhaustive search
                    matches_2D, mask = exhaustive_search.exhaustive_search(
                        query_dense_hypercolumn,
                        reference_sparse_hypercolumns,
                        Image.open(nearest_neighbor).size[::-1],
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
                        reference_filename=nearest_neighbor, #Reference filename
                        reference_2D_points=local_reconstruction.points_2D[mask],
                        reference_keypoints=None)

                    # If PnP failed, fall back to nearest-neighbor prediction
                    
                    if not prediction.success:
                        prediction = self._nearest_neighbor_prediction(
                            nearest_neighbor)
                        if prediction:
                            # Add full matches to run FeatureMetricPnP on top of NN
                            prediction = prediction._replace(num_matches=points_2D.shape[0],
                            num_inliers=0,
                            reference_inliers=local_reconstruction.points_2D[mask],
                            query_inliers=np.squeeze(points_2D),
                            points_3d = points_3D)
                            predictions.append(prediction)
                    else:
                        predictions.append(prediction)
                    if self._cache_results:
                        local_res= {
                        'reference_filename' : prediction.reference_filename,
                        'success': prediction.success,
                        'query_2D' : points_2D,
                        'reference_2D': local_reconstruction.points_2D[mask],
                        'points_3D': points_3D,
                        'num_matches': prediction.num_matches,
                        'num_inliers':prediction.num_inliers,
                        'inlier_mask':prediction.inlier_mask,
                        'quaternion':prediction.quaternion,
                        'matrix':prediction.matrix,
                        }
                        cache_dict[query_image + ":" + prediction.reference_filename] = local_res
                else:
                    # key = query_image + ":" + nearest_neighbor
                    # f_data = file_dict[key]
                    print("running only optimization")
                    key = query_image
                    f_data = file_dict[key]
                    prediction = solve_pnp.Prediction(
                                success=f_data['success'],
                                num_matches=f_data['num_matches'],
                                num_inliers=f_data['num_inliers'],
                                reference_inliers=f_data['reference_2D'][f_data['inlier_mask']],
                                query_inliers=f_data['query_2D'][f_data['inlier_mask']],
                                points_3d=f_data['points_3D'][f_data['inlier_mask']],
                                quaternion=f_data['quaternion'],
                                matrix=f_data['matrix'],
                                reference_filename=f_data['reference_filename'],
                                reference_keypoints=None,
                                inlier_mask = f_data['inlier_mask'])
                    if len(predictions) == 0:
                        predictions.append(prediction)

            if len(predictions):
                if (self._cache_results and not self._only_optimization):
                    cache_filename = self._output_path + "cached_matches/" + self._cache_filename
                    os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
                    np.savez(cache_filename, **cache_dict)

                export, best_prediction = self._choose_best_prediction(
                    predictions, query_image)
                #please note we are appending the pose twice into the output
                # before submission remember to remove all the odd raws
                output.append(export.copy())
                if best_prediction.success:
                    
                    print("running optimization for query = {} and reference = {}".format(query_image,
                                                                    best_prediction.reference_filename) )
                    st = time.time()  
                    t, quaternion, model = optimize_feature_pnp(query_dense_hypercolumn.view(channels, width, height)[None, ...],
                                        net = self._network, prediction = best_prediction, K = intrinsics, track = self.track,
                                        features = self.features)
                    en = time.time()
                    print("Time for optimization is : ", en - st)
                    # track file
                    if self.track:
                        track_pickle_path=self._output_path+"track_"+str(cnt)+".p"
                        pickle.dump(model.track_, open(track_pickle_path,"wb"))
                    else:
                        track_pickle_path = None

                    export[1:5], export[5:] = quaternion, t
                    # Add row to df with track file
                    result_frame.loc[cnt] = [best_prediction.reference_filename, query_image, best_prediction.num_matches, model.best_num_inliers_, model.initial_cost_.item(), model.best_cost_.item(), track_pickle_path]
                else:
                    result_frame.loc[cnt] = [best_prediction.reference_filename, query_image, None, None, None, None, None]
                    print("RANSAC PnP failed for {}, and we predicted pose for nearest reference image {}.".format(query_image, reference_filename))
                
                if self._log_images:
                    if np.ndim(np.squeeze(best_prediction.query_inliers)):
                        self._plot_inliers(
                            left_image_path=query_image,
                            right_image_path=best_prediction.reference_filename,
                            left_keypoints=keypoint_association.kpt_to_cv2(
                                best_prediction.query_inliers),
                            right_keypoints=keypoint_association.kpt_to_cv2(
                                best_prediction.reference_inliers),
                            matches=[(i, i) for i in range(best_prediction.num_inliers)],
                            title='Sparse-to-Dense Correspondences',
                            export_filename=self._dataset.output_converter(query_image))

                    plot_correspondences.plot_image_retrieval(
                        left_image_path=query_image,
                        right_image_path=best_prediction.reference_filename,
                        title='Best match',
                        export_filename=self._dataset.output_converter(query_image))

                output.append(export)
                tqdm_bar.set_description(
                    "[{} inliers]".format(best_prediction.num_inliers))
                tqdm_bar.refresh()
        result_frame.to_csv(self._output_path + self._output_csvname, sep=";")

        return output

    def save(self, predictions: List):
        """Export the predictions as a .txt file.

        Args:
            predictions: The list of predictions, where each line contains a
                [query name, quaternion, translation], as per the CVPR Visual
                Localization Challenge.
        """

        print('>> Saving final predictions as {}'.format(self._output_filename))
        df = pd.DataFrame(np.array(predictions[1::2]))
        df.to_csv(self._output_path + self._output_filename, sep=' ', header=None, index=None)

        print('>> Saving initial predictions as {}'.format(self._output_filename.replace(".txt", "_initial.txt")))
        df = pd.DataFrame(np.array(predictions[::2]))
        df.to_csv(self._output_path + self._output_filename.replace(".txt", "_initial.txt"), sep=' ', header=None, index=None)
        #     print('>> Saving initial predictions as {}'.format(self._output_filename.replace(".txt", "_initial.txt")))
        #     df = pd.DataFrame(np.array(predictions))
        #     df.to_csv(self._output_path + self._output_filename.replace(".txt", "_initial.txt"), sep=' ', header=None, index=None)
    # @property
    def dataset(self):
        return self._dataset
