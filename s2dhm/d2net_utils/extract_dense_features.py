import argparse

import numpy as np

# import imageio

import torch

from tqdm import tqdm

import scipy
import scipy.io
import scipy.misc

from lib.model_test import D2Net
from lib.utils import preprocess_image
from s2dhm.d2net_utils.dense_pyramid import process_multiscale

# for s2dhm integration
from PIL import Image, ImageFile
import gin
from typing import List

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
def imresize(image, image_size: int, preserve_ratio=False):
    """Resize image, while optionally preserving image ratio.
    """
    if preserve_ratio:
        image.thumbnail((image_size, image_size), Image.ANTIALIAS)
        return image
    else:
        return image.resize((image_size, image_size), Image.ANTIALIAS)

def images_from_list(image_paths, resize, preprocessing, image_size):
    if resize:
        images = [imresize(image=Image.open(i), image_size=image_size,
            preserve_ratio=True) for i in image_paths]
    else:
        images = [Image.open(i) for i in image_paths]
    images = [preprocess_image(np.array(i.convert('RGB')), preprocessing) for i in images]

    return np.array(images)

@gin.configurable
def extract_dense_features(image_paths : List[str], preprocessing : str, image_size : int, multiscale = False,
                            resize = True, model_file = 'models/d2_tf.pth', use_relu = True, to_cpu = False):

    # Creating CNN model
    model = D2Net(
        model_file=model_file,
        use_relu=use_relu,
        use_cuda=use_cuda
    )

    images = images_from_list(image_paths, resize, preprocessing, image_size)
    
    assert(len(images) == len(image_paths)) # we should have all the images

    with torch.no_grad():
        if multiscale:
            dense_descriptors = process_multiscale(
                torch.tensor(
                    images.astype(np.float32),
                    device=device
                ),
                model
            )
        else:
            dense_descriptors = process_multiscale(
                torch.tensor(
                    images.astype(np.float32),
                    device=device
                ),
                model,
                scales=[1]
            )
    if to_cpu:
        dense_descriptors = dense_descriptors.cpu().data.numpy()

    return dense_descriptors