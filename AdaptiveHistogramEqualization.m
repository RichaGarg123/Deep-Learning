clc;
close all;
clear all;

%original image
A=imread('37073.jpg');
R=A(:,:,1);
G=A(:,:,2);
B=A(:,:,3);
subplot(2,4,1)
imshow(A)
title('Original Image-37073.jpg')

subplot(2,4,2)
imhist(R, 64)
title('Red Channel')

subplot(2,4,3)
imhist(G,64)
title('Green Channel')

subplot(2,4,4)
imhist(B,64)
title('Blue Channel')

%After applying Adaptive Histogram Equalization (CLAHE)
R=adapthisteq(R);
G=adapthisteq(G);
B=adapthisteq(B);
%combining channels back
A=cat(3,R,G,B);
%plotting
subplot(2,4,5)
imshow(A)
title('Original Image after CLAHE-37073.jpg')

subplot(2,4,6)
imhist(R,64)
title('Red Channel after CLAHE')

subplot(2,4,7)
imhist(G,64)
title('Green Channel after CLAHE')

subplot(2,4,8)
imhist(B,64)
title('Blue Channel after CLAHE')
