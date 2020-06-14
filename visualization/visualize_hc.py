import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

def visualize_hc(reference_hc, query_hc_map, detection_point, output_filepath, query_img, reference_img):
    '''
    args:
    @reference_hc : reference hypercolumn of shape 1xchannels
    @query_hc_map: query hypercolumns of shape channels x width x height
    @detection_point : matching point to reference_hc
    @output_filepath: where to save the image
    '''
    f, axarr = plt.subplots(1,4) 
    channels, width, height = query_hc_map.shape
    corr_map = torch.mm(reference_hc, query_hc_map.view(channels, -1)).view(width, height, 1).numpy()
    corr_map = ((corr_map - np.amin(corr_map))/(np.amax(corr_map) - np.amin(corr_map)))
    # corr_map = np.concatenate([corr_map, corr_map, corr_map], axis = 2)
    # cv2.circle(corr_map, tuple(detection_point), 2, (0.5, 0.5, 0.0), 3)
    

    if output_filepath:
        axarr[0].imshow(reference_img)
        axarr[0].axis('off')
        axarr[1].imshow(query_img)
        axarr[1].axis('off')
        axarr[2].imshow(corr_map[:,:,0])
        axarr[2].axis('off')
        cv2.circle(corr_map, tuple(detection_point), 2, 0.1, 2)
        axarr[3].imshow(corr_map[:, :, 0])
        axarr[3].axis('off')
        # plt.imshow(corr_map[:,:,0])
        # plt.imsave(output_filepath, f)
        plt.savefig(output_filepath, dpi = 500)
    else:
        cv2.circle(corr_map, tuple(detection_point), 2, 0.1, 2)
        plt.imshow(corr_map[:,:,0])
        plt.show()


def visualize_hcq(reference_hc, query_hc_map, detection_point, output_filepath, query_img, reference_img):
    '''
    args:
    @reference_hc : reference hypercolumn of shape 1xchannels
    @query_hc_map: query hypercolumns of shape channels x width x height
    @detection_point : matching point to reference_hc
    @output_filepath: where to save the image
    '''
    #f, axarr = plt.subplots(1,4) 
    channels, width, height = query_hc_map.shape
    corr_map = torch.mm(reference_hc, query_hc_map.view(channels, -1)).view(width, height, 1).numpy()
    corr_map = ((corr_map - np.amin(corr_map))/(np.amax(corr_map) - np.amin(corr_map)))
    # corr_map = np.concatenate([corr_map, corr_map, corr_map], axis = 2)
    # cv2.circle(corr_map, tuple(detection_point), 2, (0.5, 0.5, 0.0), 3)
    #cv2.circle(corr_map, tuple(detection_point), 2, 0.1, 2)
   # plt.imshow(corr_map[:,:,0])
    plt.imsave(output_filepath,corr_map[:,:,0], dpi = 500)

if __name__ == "__main__":
    r = torch.randn(1,10)
    q = torch.randn(10, 512, 512)
    dp = (256,256)
    query_img = np.zeros((512, 512), np.float32)
    reference_img = np.zeros((512, 512), np.float32)
    visualize_hcq(r, q, dp, 'sample.png', query_img, reference_img)    

