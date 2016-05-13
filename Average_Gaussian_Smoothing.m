clc;
close all;
clear all;

%average/mean smoothing
%reading the images
A=imread('12084.jpg');
%storing the three filter sizes
h = fspecial('average',[3,3]);
j=fspecial('average',[10,10]);
k=fspecial('average',[12,20]);


%applying filters
% Applying Average Filtering
Y=imfilter(A,h);
X=imfilter(A,j);
Z=imfilter(A,k);
%plotting results for image 12084.jpg
subplot(2,3,2)
imshow(A)
title('Original Image-without smoothing')
subplot(2,3,4)
imshow(Y)
title('Average smoothing- 3x3')
subplot(2,3,5)
imshow(X)
title('Average smoothing- 10x10')
subplot(2,3,6)
imshow(Z)
title('Average smoothing- 12x20')

%Applying Gaussian Filtering
blur1=imgaussfilt(A,3); %as filter size is increased and applied image will become more and more blurred
blur2=imgaussfilt(A,[10 10]);
blur3=imgaussfilt(A,[12 20]);
subplot(2,3,2)
imshow(A)
title('Original Image-without smoothing')
subplot(2,3,4)
imshow(blur1)
title('Gaussian smoothing- 3x3')
subplot(2,3,5)
imshow(blur2)
title('Gaussian smoothing- 10x10')
subplot(2,3,6)
imshow(blur3)
title('Gaussian smoothing- 12x20')
