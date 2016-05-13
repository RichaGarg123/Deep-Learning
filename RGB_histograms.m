clc;
close all;
clear all;
%Split into RGB Channels
%all three in one plot
A=imread('3096.jpg');
R=A(:,:,1);
G=A(:,:,2);
B=A(:,:,3);
subplot(2,3,2)
imshow(A)
title('Original Image-3096.jpg')

subplot(2,3,4)
imhist(R,64)
title('Red Channel')

subplot(2,3,5)
imhist(G,64)
title('Green Channel')

subplot(2,3,6)
imhist(B,64)
title('Blue Channel')
