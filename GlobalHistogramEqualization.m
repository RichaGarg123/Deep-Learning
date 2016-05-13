clc;
close all;
clear all;
%original image
A=imread('3096.jpg');
R=A(:,:,1);
G=A(:,:,2);
B=A(:,:,3);
subplot(2,4,1)
imshow(A)
title('Original Image-3096.jpg')

subplot(2,4,2)
imhist(R, 64)
title('Red Channel-without GHE')

subplot(2,4,3)
imhist(G,64)
title('Green Channel-without GHE')

subplot(2,4,4)
imhist(B,64)
title('Blue Channel-without GHE')
%after applying GHE
R=histeq(R);
G=histeq(G);
B=histeq(B);
%combining channels back
A=cat(3,R,G,B);
%plotting
subplot(2,4,5)
imshow(A)
title('Original Image after GHE-3096.jpg')

subplot(2,4,6)
imhist(R,64)
title('Red Channel after GHE')

subplot(2,4,7)
imhist(G,64)
title('Green Channel after GHE')

subplot(2,4,8)
imhist(B,64)
title('Blue Channel after GHE')
