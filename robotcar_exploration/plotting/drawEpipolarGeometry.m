function drawEpipolarGeometry(img1, img2, x1s, x2s, F, fig1, fig2)

% IMG-1

figure(fig1); imshow(img1, []); hold on;

plot(x1s(1,:), x1s(2,:), '*g');

for i=1:size(x2s,2)
    drawEpipolarLines(F'*x2s(:,i), img1);    
end

% IMG-2

pause(0.5); 

figure(fig2); imshow(img2, []); hold on;

plot(x2s(1,:), x2s(2,:), '*g');

for i=1:size(x1s,2)
    drawEpipolarLines(F*x1s(:,i), img2);    
end

end