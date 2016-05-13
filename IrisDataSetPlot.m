clc;
close all;
clear all;
%2 classes have been considered-Iris Setosa and Iris Versicolor as these
%are linearly separable
% class label is 1 for Iris-Setosa, 0 for iris-versicolor
% have demonstrated how to take plots between Petal Length and Petal Width.
% By appropriately picking the matrices plots between other features can be
% generated too.
iris_train = xlsread('iris.xlsx','iris_train');
for i=1:70
    SepalLength(i)=iris_train(i,1);
    SepalWidth(i)=iris_train(i,2);
    PetalLength(i)=iris_train(i,3);
    PetalWidth(i)=iris_train(i,4);
    ClassLabel(i)=iris_train(i,5);
end
figure
% Plot first class
scatter(PetalLength(ClassLabel == 1), PetalWidth(ClassLabel == 1), 150, 'b', 'o')
% Plot second class.
hold on;
scatter(PetalLength(ClassLabel == 0), PetalWidth(ClassLabel ==0), 120, 'r', '*')
% scatter(SepalLength, SepalWidth, 100 , ClassLabel, ['red'; 'green']);
title('Petal Length vs Petal Width')
xlabel('Petal Length')
ylabel('Petal Width')
legend('Iris-Setosa','Iris-Versicolor')
iris_test= xlsread('iris.xlsx', 'iris_test');
