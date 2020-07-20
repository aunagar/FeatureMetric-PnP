# RobotCar Exploration

### Requirements
* [VLFeat](http://www.vlfeat.org/index.html)
* RobotCar Dataset [SDK](https://github.com/ori-mrg/robotcar-dataset-sdk.git)

### Objective
In this code, we examine the quality of the ground truth poses of the [RobotCar-Seasons](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/) and [RobotCar](https://robotcar-dataset.robots.ox.ac.uk) datasets using epipolar geometry.

### Evaluation Approach
Given a pair of two images and the corresponding ground truth poses, we directly compute the fundamental matrix, and utilize it in order to plot the epipolar lines on the images.
We also compare against the epipolar lines generated when matching SIFT features and using RANSAC to compute the fundamental matrix of the associated image pair.

### Conclusion
We found that the ground truth poses in both datasets are mostly consistent.
However, sometimes there are inaccuracies, which often appear as shifts in the vertical axis.

### Directory Structure
The directory structure is the following:
```
├── compute_fund_mat_from_poses.m   ---> Compute Fundamental Matrix from Poses
├── explore_robotcar_new.m          ---> RobotCar Exploration Script
├── explore_robotcar_seasons.m      ---> RobotCar-Seasons Exploration Script
├── images                          ---> Example Plots
│   ├── explore_robotcar_new
│   │   ├── ground_truth_a.jpg
│   │   ├── ground_truth_b.jpg
│   │   ├── sift_ransac_a.jpg
│   │   └── sift_ransac_b.jpg
│   └── explore_robotcar_seasons
│       ├── ground_truth_a.jpg
│       ├── ground_truth_b.jpg
│       ├── sift_ransac_a.jpg
│       └── sift_ransac_b.jpg
├── plotting
│   ├── drawCameras.m
│   ├── drawEpipolarGeometry.m
│   ├── drawEpipolarLines.m
│   ├── showFeatureMatchesIO.m
│   └── showFeatureMatches.m
├── quat2rotm.m                     ---> Quaternion to Rotation Matrix Conversion
├── ransac
│   ├── fundmatrix.m
│   ├── projmatrix.m
│   ├── ransacfitfundmatrix.m
│   ├── ransacfitprojmatrix.m
│   └── ransac.m
├── README.md
├── set_path.m                      ---> Sets MATLAB Path (Should be run first)
├── sfm
│   ├── decomposeE.m
│   └── linearTriangulation.m
├── timestamps_robotcar_new.txt     ---> Example Timestamps of Test Images of RobotCar-New
├── timestamps_robotcar_seasons.txt ---> Example Timestamps of Test Images of RobotCar-Seasons
└── utils
    ├── hnormalise.m
    ├── makehomogeneous.m
    ├── makeinhomogeneous.m
    ├── normalise2dpts.m
    └── normalise3dpts.m
```
