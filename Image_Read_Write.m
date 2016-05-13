clc;
close all;
clear all;
%load file
A=imread('8023.jpg');
%now coverting this to grayscale
C=rgb2gray(A);
%writing RGB image 
imwrite(A,'D:/Q1a_RGB.png');
%writing  Grayscale image
imwrite(C,'D:/Q1a_GrayScale.png');
