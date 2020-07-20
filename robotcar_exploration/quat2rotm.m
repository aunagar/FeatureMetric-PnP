function R = quat2rotm(q, tol)
    if (length(q) ~= 4)
        error('quat2rotm: %s', WBM.wbmErrorMsg.WRONG_VEC_DIM);
    end
    q = reshape(q, 4, 1);

%    qnorm = norm(q);
%    if (qnorm > 1)
%        q = q./qnorm; % normalize
%    end
    q_0 = q(1);
    q_1 = q(2);
    q_2 = q(3);
    q_3 = q(4);

    n = q' * q;
    if ( n < tol )
        R = eye(3);
        return
    end    

    q = q * sqrt(2.0/n);
    q = q * q';

    R = [1.0-q(2+1, 2+1)-q(3+1, 3+1),     q(1+1, 2+1)-q(3+1, 0+1),     q(1+1, 3+1)+q(2+1, 0+1);
             q(1+1, 2+1)+q(3+1, 0+1), 1.0-q(1+1, 1+1)-q(3+1, 3+1),     q(2+1, 3+1)-q(1+1, 0+1);
             q(1+1, 3+1)-q(2+1, 0+1),     q(2+1, 3+1)+q(1+1, 0+1), 1.0-q(1+1, 1+1)-q(2+1, 2+1)];

end
