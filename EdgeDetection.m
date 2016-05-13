clc;
close all;
clear all;
% %read image
A=imread('24077.jpg');
%convert to grayscale
X=rgb2gray(A);
%edge detection methods

%inbuilt method of Canny
C=edge(X,'canny');

%inbuilt method of Prewitt
D=edge(X,'prewitt');

%inbuilt method of Sobel
E=edge(X,'sobel');


%%%%%%%%%%%%%%%%%%%%% Fuzzy logic Implementation %%%%%%%%%%%%%%%%%%%%%%%%%%
%read image%
I=imread('37073.jpg');
I=rgb2gray(I);
%scaling all the pixel values between zero and one%
I=im2double(I);
Gx = [-1 1];
Gy = Gx';
Ix = conv2(I,Gx,'same');
Iy = conv2(I,Gy,'same');

edgeFIS = newfis('edgeDetection');
edgeFIS = addvar(edgeFIS,'input','Ix',[-1 1]);
edgeFIS = addvar(edgeFIS,'input','Iy',[-1 1]);

sx = 0.1; sy = 0.1;
edgeFIS = addmf(edgeFIS,'input',1,'zero','gaussmf',[sx 0]);
edgeFIS = addmf(edgeFIS,'input',2,'zero','gaussmf',[sy 0]);

edgeFIS = addvar(edgeFIS,'output','Iout',[0 1]);
wa = 0.1; wb = 1; wc = 1;
ba = 0; bb = 0; bc = .7;
edgeFIS = addmf(edgeFIS,'output',1,'white','trimf',[wa wb wc]);
edgeFIS = addmf(edgeFIS,'output',1,'black','trimf',[ba bb bc]);
r1 = 'If Ix is zero and Iy is zero then Iout is white';
r2 = 'If Ix is not zero or Iy is not zero then Iout is black';
r = char(r1,r2);
edgeFIS = parsrule(edgeFIS,r);
showrule(edgeFIS)

Ieval = zeros(size(I));% Preallocate the output matrix
for ii = 1:size(I,1)
    Ieval(ii,:) = evalfis([(Ix(ii,:));(Iy(ii,:));]',edgeFIS);
end
imwrite(~Ieval,'untitled2.png');
figure; image(I,'CDataMapping','scaled'); colormap('gray');
title('Original Grayscale Image')

figure; image(Ieval,'CDataMapping','scaled'); colormap('gray');
title('Edge Detection Using Fuzzy Logic')
% 
% 
%%%%%%%%%%%%%%%%%%%%% Sobel- Implementation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A=imread('37073.jpg');
B=rgb2gray(A);

C=double(B);


for i=1:size(C,1)-2
    for j=1:size(C,2)-2
        %Sobel mask for x-direction:
        Gx=((2*C(i+2,j+1)+C(i+2,j)+C(i+2,j+2))-(2*C(i,j+1)+C(i,j)+C(i,j+2)));
        %Sobel mask for y-direction:
        Gy=((2*C(i+1,j+2)+C(i,j+2)+C(i+2,j+2))-(2*C(i+1,j)+C(i,j)+C(i+2,j)));
      
        %The gradient of the image
        %B(i,j)=abs(Gx)+abs(Gy);
        B(i,j)=sqrt(Gx.^2+Gy.^2);
      
    end
end
% imwrite(B,'untitled1.png')
figure,imshow(B); title('Sobel gradient');
% threshold value
Thresh=0.5; %lower the threshold more edges will be detected and algo will become susceptible to noises.
B=max(B,Thresh);
B(B==round(Thresh))=0;
B=uint8(B);
imwrite(B,'Sobel_Edges.png')

