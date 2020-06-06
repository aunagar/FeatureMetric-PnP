import torch
import torch.nn as nn
import torch.nn.functional as F

def process_multiscale(image, model, scales=[.5, 1, 2]):
    b, _, h_init, w_init = image.size()
    device = image.device
    assert(b == 1)

    all_keypoints = torch.zeros([3, 0])
    all_descriptors = torch.zeros([
        model.dense_feature_extraction.num_channels, 0
    ])
    all_scores = torch.zeros(0)

    previous_dense_features = None
    banned = None
    for idx, scale in enumerate(scales):
        current_image = F.interpolate(
            image, scale_factor=scale,
            mode='bilinear', align_corners=True
        )
        _, _, h_level, w_level = current_image.size()

        dense_features = model.dense_feature_extraction(current_image)
        del current_image

        _, _, h, w = dense_features.size()

        # Sum the feature maps.
        if previous_dense_features is not None:
            dense_features += F.interpolate(
                previous_dense_features, size=[h, w],
                mode='bilinear', align_corners=True
            )
            del previous_dense_features

        dense_features = F.normalize(dense_features[0], dim=0)

        previous_dense_features = dense_features
        del dense_features
    return previous_dense_features[None, ...]
