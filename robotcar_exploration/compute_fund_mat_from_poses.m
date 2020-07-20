function [F] = compute_fund_mat_from_poses(P1, P2, K, G_camera_image)

%    P1 = P1 * inv(G_camera_image);
%    P2 = P2 * inv(G_camera_image);

    % Compute Relative Pose
     %P = P1 \ P2;
     %P = P2 \ P1;
     %P = P2*inv(P1);
     P = P1*inv(P2);
     [R, t] = deal( P(1:3,1:3), P(1:3,end) );

%    [R1, t1] = deal( P1(1:3,1:3), P1(1:3,end) );
%    [R2, t2] = deal( P2(1:3,1:3), P2(1:3,end) );
%    R = R1 \ R2
%    t = t1 - (R \ t2)
%    t = (R \ t2) - t1
    %t = R \ (t2 - t1)


    % Compute Fundamental Matrix
    F = inv(K)'*skew(t)*R*inv(K);
end

function S = skew(a)
    S = [0 -a(3) a(2);
        a(3) 0 -a(1);
        -a(2) a(1) 0];
end
