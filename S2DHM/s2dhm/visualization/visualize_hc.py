import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

def visualize_hc(reference_hc, query_hc_map, detection_point, output_filepath):
    '''
    args:
    @reference_hc : reference hypercolumn of shape 1xchannels
    @query_hc_map: query hypercolumns of shape channels x width x height
    @detection_point : matching point to reference_hc
    @output_filepath: where to save the image
    '''

    channels, width, height = query_hc_map.shape
    corr_map = torch.mm(reference_hc, query_hc_map.view(channels, -1)).view(width, height, 1).numpy()
    corr_map = ((corr_map - np.amin(corr_map))/(np.amax(corr_map) - np.amin(corr_map)))
    # corr_map = np.concatenate([corr_map, corr_map, corr_map], axis = 2)
    # cv2.circle(corr_map, tuple(detection_point), 2, (0.5, 0.5, 0.0), 3)
    cv2.circle(corr_map, tuple(detection_point), 2, 0.1, 2)

    if output_filepath:
        plt.imsave(output_filepath, corr_map[:,:,0])
    else:
        plt.imshow(corr_map)
        plt.show()

if __name__ == "__main__":
    r = torch.randn(1,10)
    q = torch.randn(10, 512, 512)
    dp = (256,256)
    visualize_hc(r, q, dp, 'sample.png')    

