clc;
close all;
clear all;

%reading train-images.idx3
fid1 = fopen('train-images.idx3-ubyte','r','ieee-be');
A(1) = fread(fid1,1,'uint32');
A(2) = fread(fid1,1,'uint32');
A(3) = fread(fid1,1,'uint32');
A(4) = fread(fid1,1,'uint32');
%28*28*60000=47040000
for i=1:47040000
    A(i)=fread(fid1,1,'unsigned char');
end
%now rearranging the data in the desired format
k=1;
for i=1:60000
    for j=1:784
        TrainData(i,j)=A(k);
        k=k+1;
    end
end

%reading train-labels.idx1
fid2 = fopen('train-labels.idx1-ubyte','r','ieee-be');
B(1) = fread(fid2,1, 'uint32');
B(2) = fread(fid2,1, 'uint32');
for i=1:60000
    B(i)=fread(fid2,1,'unsigned char');
end
%now rearranging the data in the desired format
TrainLabels=B.';

%reading t10k-images.idx3
fid3 = fopen('t10k-images.idx3-ubyte','r','b');
C(1) = fread(fid3,1,'uint32');
C(2) = fread(fid3,1,'uint32');
C(3) = fread(fid3,1,'uint32');
C(4) = fread(fid3,1,'uint32');
%28*28*10000
for i=1:7840000
    C(i)=fread(fid3,1,'unsigned char');
end
%now rearranging the data in the desired format
k=1;
for i=1:10000
    for j=1:784
        TestData(i,j)=C(k);
        k=k+1;
    end
end


%reading t10k-labels.idx1
fid4 = fopen('t10k-labels.idx1-ubyte','r','ieee-be');
D(1) = fread(fid4,1,'uint32');
D(2)=fread(fid4,1,'uint32');
for i=1:10000
    D(i)=fread(fid4,1,'unsigned char');
end
%now rearranging the data in the desired format
TestLabels=D.'
