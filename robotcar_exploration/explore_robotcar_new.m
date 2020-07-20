% =========================================================================
% RobotCar Exploration
% =========================================================================
clear; clc; close all; rng(1994);

% Initialize VLFeat (http://www.vlfeat.org/)
run('../vlfeat-0.9.21/toolbox/vl_setup.m');
% Setup paths to SDK and RobotCar dataset slice
addpath('../sdk/matlab/');
models_dir = '../sdk/models/';
extrinsics_dir = '../sdk/extrinsics/';
image_dir = '../dataset/2015-10-30-11-56-36/mono_rear/'; 
rtk_file = '../dataset/rtk/2015-10-30-11-56-36/rtk.csv';

[fx, fy, cx, cy, G_camera_image, LUT] = ReadCameraModel(image_dir, models_dir);

K = [fx 0 cx;
     0 fy cy;
     0  0  1];

 K=K*G_camera_image(1:3,1:3);

timestamps = dlmread('timestamps_robotcar_new.txt');

% Choose image pair (timestamp) index
id = 20;
timestamps = timestamps(id:id+1,:);

% load ground truth (interpolated) poses ( rtk_poses )
%rtk_poses = InterpolatePoses(rtk_file, str2num(timestamps)', str2num(timestamps(1, :)), true);
rtk_poses = InterpolatePoses(rtk_file, timestamps', timestamps(1, :), true);

img1 = single( rgb2gray( LoadImage(image_dir, timestamps(1,  :), LUT) ) );
img2 = single( rgb2gray( LoadImage(image_dir, timestamps(end,:), LUT) ) );

% extract SIFT features and match
[fa, da] = vl_sift( img1 );
[fb, db] = vl_sift( img2 );

% don't take features at the top of the image - only background
sel = fa(2,:) > 100;
fa = fa(:,sel);
da = da(:,sel);

[matches, ~] = vl_ubcmatch(da, db);

% plot initial matches
%showFeatureMatches(img1, fa(1:2, matches(1,:)), img2, fb(1:2, matches(2,:)), 20);

xa = makehomogeneous( fa(1:2, matches(1,:)) );
xb = makehomogeneous( fb(1:2, matches(2,:)) );

% compute fundamental matrix F [ s.t. xb' * F * xa = 0 ] and respective inliers using 8-point ransac
[F, inliers] = ransacfitfundmatrix( xa, xb, 1e-4 );

outliers = setdiff( 1:size(matches,2), inliers );

fai = fa(:, matches(1,inliers));
dai = da(:, matches(1,inliers));
xai = xa(:, inliers); 
xao = xa(:, outliers);
xbi = xb(:, inliers); 
xbo = xb(:, outliers);

% plot inlier and outlier matches
%pause(0.5); showFeatureMatches(img1, xai(1:2,:), img2, xbi(1:2,:), 21); % plot inlier matches
%pause(0.5); showFeatureMatches(img1, xao(1:2,:), img2, xbo(1:2,:), 22); % plot outlier matches
% pause(0.5); showFeatureMatchesIO(img1, xai(1:2,:), xao(1:2,:), ... 
%                                  img2, xbi(1:2,:), xbo(1:2,:), 23);

% compute fundamental matrix from ground truth poses
F2 = compute_fund_mat_from_poses(rtk_poses{1}, rtk_poses{2}, K, G_camera_image);

% plot epipolar lines
pause(0.5); drawEpipolarGeometry( img1, img2, xai(:,1:10:end), xbi(:,1:10:end), F, 23,24 );
pause(0.5); drawEpipolarGeometry( img1, img2, xai(:,1:10:end), xbi(:,1:10:end), F2, 25,26 );
