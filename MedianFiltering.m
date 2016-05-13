clc;
close all;
clear all;
%Denoising- Median filtering
% Here image considered is a RGB image.
J=imread('3.jpg');
%1.Applying Median Filtering on RGB image
%This is done by separating three channels of RGB image.
%separating channels of RGB image
R=J(:,:,1);
G=J(:,:,2);
B=J(:,:,3);
%applying median filtering on the 3 channels
X=medfilt2(R,[5,5]);
Y=medfilt2(G,[5,5]);
Z=medfilt2(B,[5,5]);
%merging channels back together to image
finalImage=cat(3,X,Y,Z);
subplot(1,2,1)
imshow(J)
title('Original Image')
subplot(1,2,2)
imshow(finalImage)
title('After Median Filtering-Filter size:8')

%Applying median filtering on a Grayscale image.
%Converting the given image to Grayscale from RGB
J=rgb2gray(J);
B = medfilt2(J, [5,5]); % as window size increases, image becomes more blurred and edges are well maintained
%but image has been smoothened to remove the noise
figure
imshow(B)
title('Median Filtering')

%in case noise needs to be introduced into an image- opposite of denoising
J=imread('sample.jpg');
X = imnoise(J,'salt & pepper',0.02);


