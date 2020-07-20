% show feature matches between two images (inliers and outliers colored differently)
%
% Input:
%   img1        - n x m color image 
%   in1         - 2 x k matrix, holding inlier keypoint coordinates of first image
%   out1        - 2 x k matrix, holding outlier keypoint coordinates of first image
%   img2        - n x m color image 
%   in2         - 2 x k matrix, holding inlier keypoint coordinates of second image
%   out2        - 2 x k matrix, holding outlier keypoint coordinates of second image
%   fig         - figure id
function showFeatureMatchesIO(img1, in1, out1, img2, in2, out2, fig)
    [sx, sy, sz] = size(img1);
    img = [img1, img2];
    
    in2  = in2  + repmat([sy, 0]', [1, size(in2, 2)]);
    out2 = out2 + repmat([sy, 0]', [1, size(out2, 2)]);
    
    figure(fig), imshow(img, []);    
    hold on, plot(out1(1,:), out1(2,:), '+r', 'LineWidth', 2);    
    hold on, plot(out2(1,:), out2(2,:), '+r', 'LineWidth', 2);
    hold on, plot(in1 (1,:), in1 (2,:), '+g', 'LineWidth', 2);
    hold on, plot(in2 (1,:), in2 (2,:), '+g', 'LineWidth', 2);
    hold on, plot([out1(1,:); out2(1,:)], [out1(2,:); out2(2,:)], 'w');
    hold on, plot([in1(1,:); in2(1,:)], [in1(2,:); in2(2,:)], 'b');
end