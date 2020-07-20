% =========================================================================
% RobotCar Exploration
% =========================================================================
clear; clc; close all; rng(1994);

% Initialize VLFeat (http://www.vlfeat.org/)
run('../vlfeat-0.9.21/toolbox/vl_setup.m');
path_to_images = '../RobotCar-Seasons/overcast-reference/rear/';
nvm_file = '../RobotCar-Seasons/rear_1000-1050.nvm';

% Rear camera intrinsics
fx = 400;
fy = 400;
cx = 508.222931;
cy = 498.187378;

K = [fx 0 cx;
     0 fy cy;
     0  0  1];

% Rear camera extrinsics
%E = [-0.999802, -0.011530, -0.016233, 0.060209;
%     -0.015184, 0.968893, 0.247013, 0.153691;
%     0.012880, 0.247210, -0.968876, -2.086142;
%     0.000000, 0.000000, 0.000000, 1.000000];

% timestamps
timestamps = num2str(dlmread('timestamps_robotcar_seasons.txt'));

%i = input('i = ');
i = 4;

id1 = i;
id2 = i+1;
img1 = single( rgb2gray( imread([path_to_images timestamps(id1,:) '.jpg']) ) );
img2 = single( rgb2gray( imread([path_to_images timestamps(id2,:) '.jpg']) ) );

% Ground Truth Data
nvm_data = load(nvm_file);
quaternions = nvm_data(:,2:5);
camera_center = nvm_data(:,6:8);

% GT Poses
R1 = quat2rotm( quaternions(id1,:), 1e-10);
R2 = quat2rotm( quaternions(id2,:), 1e-10);
t1 = -R1 * camera_center(id1,:)';
t2 = -R2 * camera_center(id2,:)';

P1 = [R1 t1; 0 0 0 1];
P2 = [R2 t2; 0 0 0 1];
F2 = compute_fund_mat_from_poses(P1, P2, K, eye(4));

% extract SIFT features and match
[fa, da] = vl_sift( img1 );
[fb, db] = vl_sift( img2 );

% don't take features at the top of the image - only background
sel = fa(2,:) > 100;
fa = fa(:,sel);
da = da(:,sel);

[matches, ~] = vl_ubcmatch(da, db);

% plot initial matches
% showFeatureMatches(img1, fa(1:2, matches(1,:)), img2, fb(1:2, matches(2,:)), 20);

xa = makehomogeneous( fa(1:2, matches(1,:)) );
xb = makehomogeneous( fb(1:2, matches(2,:)) );

[F, inliers] = ransacfitfundmatrix( xa, xb, 1e-4 );

outliers = setdiff( 1:size(matches,2), inliers );
fai = fa(:, matches(1,inliers));
dai = da(:, matches(1,inliers));
xai = xa(:, inliers); xao = xa(:,outliers);
xbi = xb(:, inliers); xbo = xb(:,outliers);

% plot inlier and outlier matches
% pause(0.5); showFeatureMatches(img1, xai(1:2,:), img2, xbi(1:2,:), 21); % plot inlier matches
% pause(0.5); showFeatureMatches(img1, xao(1:2,:), img2, xbo(1:2,:), 22); % plot outlier matches
% pause(0.5); showFeatureMatchesIO(img1, xai(1:2,:), xao(1:2,:), ... 
%                                  img2, xbi(1:2,:), xbo(1:2,:), 23);

pause(0.5); drawEpipolarGeometry( img1, img2, xai(:,1:10:end), xbi(:,1:10:end), F, 23, 24 );
pause(0.5); drawEpipolarGeometry( img1, img2, xai(:,1:10:end), xbi(:,1:10:end), F2, 25, 26 );

return

% Validate compute_fund_mat_from_poses by plugin in the SIFT poses
% compute essential matrix
E = K'*F*K;

% compute projection matrices
xai_calibrated = K \ xai;
xbi_calibrated = K \ xbi;

Q1 = eye(4);
Q2 = decomposeE( E, xai_calibrated, xbi_calibrated );

F_q = compute_fund_mat_from_poses(Q1, Q2, K, eye(4))
F
