clc;
close all;
clear all;
%2 classes have been considered-Iris Setosa and Iris Versicolor as these
%are linearly separable

iris_train = xlsread('iris.xlsx','iris_train');
iris_test= xlsread('iris.xlsx', 'iris_test');
% since number of features is 4, we will generate a vector of 5 weights
%last one for bias
%these weights will be generated randomly to start with
for i=1:5
    w(i)=0;
end
%keeping track of number of iterations
num_iteration=0;
%theta is the threshold value
theta=0;
%learning rate- every time the wieghts get adjusted with the help of this
learning_rate=0.001;
%global error- sum of squared mean error
% local error- error in that iteration
localError=1;


while num_iteration<=20000
    globalError=0;
    num_iteration=num_iteration+1;
    % since this is training we will consider the 70 training instances
    for i=1:70
        sum=0;
        for j=1:4
           sum=sum+ iris_train(i,j)*w(j);
        end
        %adding the bias
        sum=sum+w(5);
        if(sum>=theta)
            output=1;
        else
            output=0;
        end
        %iris_train 5th column contains the actual label
        localError=iris_train(i,5)-output;
        %updating weights
        for k=1:4
            w(k)=w(k)+learning_rate* localError* iris_train(i,k);
        end
        w(5)=w(5)+learning_rate*localError;
        % now computing global error for this instance
        globalError=globalError+localError*localError;
    end
    % now we will have the global error for all instances in this iteration
    globalError=sqrt(globalError);
    if globalError~=0
    disp(globalError)
    end
    if globalError==0
        break;
    end
    
end

%testing the results
count=0;
for i=1:30
     sum=0;
        for j=1:4
           sum=sum+ iris_test(i,j)*w(j);
        end
        sum=sum+w(5);
        if(sum>=theta)
            output=1;
        else
            output=0;
        end
         localError=iris_test(i,5)-output;
         if localError ~=0
             count=count+1;
             disp(i);
         end
end
disp(count/30);


