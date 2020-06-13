import gin
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


@gin.configurable
def plot_image_retrieval(left_image_path: str,
                         right_image_path: str,
                         title: str,
                         export_folder: str,
                         export_filename: str):
    """Display (query, nearest-neighbor) pairs of images."""
    left_image = cv2.cvtColor(cv2.imread(left_image_path), cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(cv2.imread(right_image_path), cv2.COLOR_BGR2RGB)
    Path(export_folder, export_filename).parent.mkdir(exist_ok=True, parents=True)
    height = max(left_image.shape[0], right_image.shape[0])
    width = left_image.shape[1] + right_image.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:left_image.shape[0], 0:left_image.shape[1]] = left_image
    output[0:right_image.shape[0], left_image.shape[1]:] = right_image[:]
    fig = plt.figure(figsize=(16, 7), dpi=160)
    plt.imshow(output)
    plt.title(title)
    plt.axis('off')
    fig.savefig(str(Path(export_folder, export_filename)))
    plt.close(fig)

@gin.configurable
def plot_correspondences(left_image_path: str,
                         right_image_path: str,
                         left_keypoints: List[cv2.KeyPoint],
                         right_keypoints: List[cv2.KeyPoint],
                         matches: np.ndarray,
                         title: str,
                         export_folder: str,
                         export_filename: str):
    """Display feature correspondences."""
    left_image = cv2.cvtColor(cv2.imread(left_image_path), cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(cv2.imread(right_image_path), cv2.COLOR_BGR2RGB)
    Path(export_folder, export_filename).parent.mkdir(exist_ok=True, parents=True)
    height = max(left_image.shape[0], right_image.shape[0])
    width = left_image.shape[1] + right_image.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:left_image.shape[0], 0:left_image.shape[1]] = left_image
    output[0:right_image.shape[0], left_image.shape[1]:] = right_image[:]

    # Draw Lines and Points
    for m in matches:
        left = left_keypoints[m[0]].pt
        right = tuple(sum(x) for x in zip(
            right_keypoints[m[1]].pt, (left_image.shape[1], 0)))
        cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255), 2)

    fig = plt.figure(figsize=(16, 7), dpi=160)
    plt.imshow(output)
    plt.title(title)
    plt.axis('off')
    fig.savefig(str(Path(export_folder, export_filename)))
    plt.close(fig)

@gin.configurable
def plot_detections(left_image_path: str,
                    right_image_path: str,
                    left_keypoints: np.ndarray,
                    right_keypoints: np.ndarray,
                    title: str,
                    export_folder: str,
                    export_filename: str):
    """Display Superpoint detections."""
    left_image = cv2.cvtColor(cv2.imread(left_image_path), cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(cv2.imread(right_image_path), cv2.COLOR_BGR2RGB)
    Path(export_folder, export_filename).parent.mkdir(exist_ok=True, parents=True)

    height = max(left_image.shape[0], right_image.shape[0])
    width = left_image.shape[1] + right_image.shape[1]
    offset = left_image.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:left_image.shape[0], 0:left_image.shape[1]] = left_image
    output[0:right_image.shape[0], left_image.shape[1]:] = right_image[:]

    fig = plt.figure(figsize=(16, 7), dpi=160)
    plt.imshow(output)
    plt.scatter(
        left_keypoints.T[:, 0], left_keypoints.T[:, 1], c='red', s=5)
    plt.scatter(
        right_keypoints.T[:, 0] + offset, right_keypoints.T[:, 1], c='red', s=5)
    plt.title(title)
    plt.axis('off')
    fig.savefig(str(Path(export_folder, export_filename)))
    plt.close(fig)

def plot_all_points(left_image_path : str,
                    right_image_path : str,
                    left_keypoints: List[cv2.KeyPoint],
                    right_keypoints: List[cv2.KeyPoint],
                    is_outlier: List[bool],
                    title: str,
                    export_folder: str,
                    export_filename: str):
    """Display feature correspondences."""
    left_image = cv2.cvtColor(cv2.imread(left_image_path), cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(cv2.imread(right_image_path), cv2.COLOR_BGR2RGB)
    Path(export_folder, export_filename).parent.mkdir(exist_ok=True, parents=True)
    height = max(left_image.shape[0], right_image.shape[0])
    width = left_image.shape[1] + right_image.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:left_image.shape[0], 0:left_image.shape[1]] = left_image
    output[0:right_image.shape[0], left_image.shape[1]:] = right_image[:]

    # Draw Lines and Points
    for i in range(len(left_keypoints)):
        left = left_keypoints[i].pt
        right = tuple(sum(x) for x in zip(
            right_keypoints[i].pt, (left_image.shape[1], 0)))
        if is_outlier[i]:
            continue
            # cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0), 2)
        else:
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (50, 205, 50), 2)

    fig = plt.figure(figsize=(16, 7), dpi=160)
    plt.imshow(output)
    plt.title(title)
    plt.axis('off')
    fig.savefig(str(Path(export_folder, export_filename)))
    plt.close(fig)

def plot_points_before_and_after_optimization(
                    query_image_path : str,
                    query_keypoints_before_opt: List[cv2.KeyPoint],
                    query_keypoints_after_opt: List[cv2.KeyPoint],
                    is_outlier: List[bool],
                    title: str,
                    export_folder: str,
                    export_filename: str):
    
    query_image = cv2.cvtColor(cv2.imread(query_image_path), cv2.COLOR_BGR2RGB)
    Path(export_folder, export_filename).parent.mkdir(exist_ok=True, parents=True)

    for i in range(len(query_keypoints_before_opt)):
        pts_before = query_keypoints_before_opt[i].pt
        pts_after  = query_keypoints_after_opt[i].pt
        if is_outlier[i]:
            continue
        else:
            cv2.circle(query_image, tuple(map(int, pts_before)), 2, (255, 0, 0), 1) # Red
            cv2.circle(query_image, tuple(map(int, pts_after)), 2, (0, 255, 0), 1) # Green

    fig = plt.figure(figsize=(16, 7), dpi=1000)
    plt.imshow(query_image)
    plt.title(title)
    plt.axis('off')
    fig.savefig(str(Path(export_folder, export_filename)))
    plt.close(fig)

def plot_points_on_reference_image(
                    ref_image_path : str,
                    ref_keypoints: List[cv2.KeyPoint],
                    is_outlier: List[bool],
                    title: str,
                    export_folder: str,
                    export_filename: str):
    
    ref_image = cv2.cvtColor(cv2.imread(ref_image_path), cv2.COLOR_BGR2RGB)
    Path(export_folder, export_filename).parent.mkdir(exist_ok=True, parents=True)

    for i in range(len(ref_keypoints)):
        pts = ref_keypoints[i].pt
        if is_outlier[i]:
            continue
        else:
            cv2.circle(ref_image, tuple(map(int, pts)), 2, (0, 0, 255), 1)

    fig = plt.figure(figsize=(16, 7), dpi=1000)
    plt.imshow(ref_image)
    plt.title(title)
    plt.axis('off')
    fig.savefig(str(Path(export_folder, export_filename)))
    plt.close(fig)
