import BA.visualization.plot_3d_tools as plot_3d
import numpy as np
import cv2
import matplotlib.pyplot as plt


def create_frames_with_camera_pose(rgb_img,R_list, t_list, cost_list, point_list, dpi=160, cost_plot_height=300, mask_inlier_list = None):
    # Create a list of images, size (H+cost_plot_height)x2*W
    # Takes some time (~ half a minute with n_iters=100 and image size 1024x1024)
    # Change DPI and draw parameters(radius, thickness) as you like
    my_dpi = dpi #For plotting, change this as you like, 160 works good for 1024x1024
    n_iters = len(R_list) -1
    # Create base-image im RGB
    # rgb_img = cv2.cvtColor(img.astype('uint8').copy(),cv2.COLOR_GRAY2RGB)
    height, width, layers = rgb_img.shape

    #cost_plot_height = 300

    # Set of 2D points: n_iters+1 x N x 2
    pts = np.array(point_list,dtype=np.int32)
    img_list=[]
    for i in range(n_iters+1):
        plt.close("all")
        # Create a copy of base image to work on
        img_int = rgb_img.copy()

        label = "Cost: %f" % cost_list[i]

        # Create cost plot (top, you can add any measure vs. iteration you like here, e.g. the error of a specific feature)
        fig, ax= plt.subplots(1,1,figsize=(width*2/my_dpi, cost_plot_height/my_dpi), dpi=my_dpi)
        ax.plot(range(i+1),np.array(cost_list[:i+1]), color = "red")
        ax.set_xlim([0,n_iters+1])
        ax.set_title("Cost vs. Iterations")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Cost (mean)")
        ax.set_ylim([0,max(cost_list)*1.1])
        cost_fig_data = plot_3d.fig2data(fig)
        plt.tight_layout()


        # Visualize camera pose  in 3D (example) (right)
        fig = plt.figure(figsize=(width/my_dpi, height/my_dpi), dpi=my_dpi)
        ax = fig.add_subplot(111, projection='3d')
        # Draw Coordinate frame
        plot_3d.plot_coordinate_system(ax, size = (0.5,0.5,0.5))
        # Plot initial camera pose (green pyramid)
        # Swap to R_init, t_init if you want initial pose
        plot_3d.plot_camera(ax,R_list[0].numpy(),t_list[0].numpy(), edgecolor="g", facecolor=None, alph=0.05) 
        # Plot final camera pose (red pyramid)
        plot_3d.plot_camera(ax,R_list[i].numpy(),t_list[i].numpy(), edgecolor="r", facecolor="r", alph=0.05)
        # Adjust Angle of 3D plot view
        ax.view_init(elev=-65, azim=-90)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-0.2, 1)
        #Remove Whitespace
        fig.subplots_adjust(left=-0.2, right=1.2, bottom=0, top=1)
        # Convert image to an array so we can concatenate with our image
        fig_data = plot_3d.fig2data(fig)

        # For every 2D point in our image (left)
        for j, p in enumerate(pts):
            # Draw past line
            cv2.polylines(img_int, np.int32([pts[:i+1,j,:]]), False, (0,0,255), 1)
            # Draw start point
            cv2.circle(img_int, tuple(pts[0,j,:]), 4, (0,255,0), -1) 
            # Draw current point
            cv2.circle(img_int, tuple(pts[i,j,:]), 4, (0,0,255), -1) 
            # Draw Inliers
            if mask_inlier_list is not None and mask_inlier_list[i,j]:
                cv2.circle(img_int, tuple(pts[i,j,:]), 4, (255,0,0), -1) 
            # Draw Label
            cv2.putText(img_int, label, (20,20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)

        # Append image list
        lower_ims = np.concatenate((img_int,fig_data), axis=1)
        img_list.append(np.concatenate((cost_fig_data,lower_ims), axis=0))
    return img_list

def save_video(frames, video_path,fr=2):
    height, width, layers = frames[0].shape
    #framerate (frames/second)
    video= cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 2, (width,height))
    for image in frames:
        video.write(image)
    video.release()