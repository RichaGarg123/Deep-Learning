# -*- coding: utf-8 -*-
"""
Created on Sun May 08 06:46:06 2016

@author: RICHA
"""
# -*- coding: utf-8 -*-


import numpy as np
import struct

import os
print os.getcwd()

filename = 'C:\\Program Files\\MATLAB\\MATLAB Production Server\\R2015a\\bin\\COGS260\\HW2'


def read_image_file(data_file):
    """
    Reads the big endian Mnist file and comverts it into little endian numpy array of images.
    """
    with open(data_file,'rb') as f:
        data = f.read(4)
        magic_no = struct.unpack('>L',data)
        if magic_no==2049 or magic_no==2051:
            print 'Incorrectly parsing files'
        print ' magic no = %d '%magic_no
        data = f.read(4)
        num_data, = struct.unpack('>L',data)
        print ' Number of data points = %d '% num_data
        data = f.read(4)
        rows, = struct.unpack('>L',data)
        data = f.read(4)
        cols, = struct.unpack('>L',data)
        vec_len = rows*cols
        print ' The number of rows = %d and num of cols = %d'% (rows,cols)
        unpacked_data = np.zeros((num_data,vec_len),np.int16)
        for i in range(num_data):
            data_i = []
            for j in range(vec_len):
                temp_data, = struct.unpack('>B',f.read(1))
                unpacked_data[i,j] = temp_data

    return unpacked_data

def read_label_file(label_file):
    """
    Reads the big endian Label file and converts it into big endian 
    """
    with open(label_file,'rb') as f:
        data = f.read(4)
        magic_no = struct.unpack('>L',data)
        if magic_no==2049 or magic_no==2051:
            print 'Incorrectly parsing files'
        print ' magic no = %d '%magic_no
        data = f.read(4)
        num_data, = struct.unpack('>L',data)
        print ' Number of data points = %d '% num_data
        unpacked_data = np.zeros((num_data),np.uint8)
        for i in range(num_data):
            temp_data, = struct.unpack('>B',f.read(1))
            unpacked_data[i] = temp_data

    return unpacked_data

train_data = read_image_file('C:\\Program Files\\MATLAB\\MATLAB Production Server\\R2015a\\bin\\COGS260\\HW2\\train-images.idx3-ubyte')
train_label=read_label_file('C:\\Program Files\\MATLAB\\MATLAB Production Server\\R2015a\\bin\\COGS260\\HW2\\train-labels.idx1-ubyte')

train_dataset=train_data[0:50000,]
test_dataset=train_data[50000:60000,]
train_labels= train_label[0:50000,]
test_labels=train_label[50000:60000,]
mean_train=np.mean(train_dataset, axis=0)
sd_train=np.std(train_dataset, axis=0)
train_dataset=(train_dataset-mean_train)/(sd_train +0.00000001)
test_dataset=(test_dataset-mean_train)/(sd_train +0.00000001)

D=784
K=10
h1 = 100 # size of hidden layer 1
h2= 100 # size of hidden Layer 2
W = np.random.uniform(low=-np.sqrt(6./(D+h1)),high=np.sqrt(6./(D+h1)),size=(D,h1))
b = np.zeros((1,h1))        
W1 = np.random.uniform(low=-np.sqrt(6./(h2+h1)),high=np.sqrt(6./(h2+h1)),size=(h1,h2))
b1 = np.zeros((1,h2))
W2 = np.random.uniform(low=-np.sqrt(6./(K+h2)),high=np.sqrt(6./(K+h2)),size=(h2,K))
b2 = np.zeros((1,K))

step_size = 1e-01
reg = 1e-3 
num_examples = train_dataset.shape[0]
for i in xrange(10000):
  
  # evaluate class scores, [N x K]
  hidden_layer1 = np.maximum(0, np.dot(train_dataset, W) + b) # note, ReLU activation
  hidden_layer2=  np.maximum(0, np.dot(hidden_layer1, W1)+b1) # note, ReLU activation
  scores = np.dot(hidden_layer2, W2) + b2
  
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  
  
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples),train_labels])
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
  loss = data_loss 
  if i % 10 == 0:
    print "iteration %d: loss %f" % (i, loss)
  
  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),train_labels] -= 1
  dscores /= num_examples
  
  # backpropate the gradient to the parameters
  #backpropagation will be done in the order
  # dW2, db2
  # hidden layer 2
  #ReLU
  # dw1 db1
  #hidden layer 1
  #ReLU
  # W b
  # first backprop into parameters W2 and b2
  dW2 = np.dot(hidden_layer2.T, dscores)
  db2 = np.sum(dscores, axis=0, keepdims=True)
  # backprop into hidden layer 2
  dhiddenLayer2=np.dot(dscores,W2.T)
  #backprop the ReLU non-linearity
  dhiddenLayer2[hidden_layer2<=0]=0
  # backprop into parameters W1 and b1
  dW1=np.dot(hidden_layer1.T,dhiddenLayer2)
  db1=np.sum(dhiddenLayer2, axis=0, keepdims=True)
  #backprop into hidden layer 1
  dhiddenLayer1=np.dot(dhiddenLayer2, W1.T)
  #backprop the ReLU non-linearity
  dhiddenLayer1[hidden_layer1<=0]=0
  #backprop into W and b
  dW= np.dot(train_dataset.T, dhiddenLayer1)
  db=np.sum(dhiddenLayer1, axis=0, keepdims=True)
  
  
  # add regularization gradient contribution
  
  dW2 += reg * W2
  dW += reg * W
  
  
  # perform a parameter update
  W += -step_size * dW
  b += -step_size * db
  W1 += -step_size * dW1
  b1 += -step_size * db1
  W2 += -step_size * dW2
  b2 += -step_size * db2
  

# evaluate training set accuracy
hidden_layer1 = np.maximum(0, np.dot(train_dataset, W) + b)
hidden_layer2=np.maximum(0,np.dot(hidden_layer1, W1)+b1)
scores = np.dot(hidden_layer2, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print 'training accuracy: %.2f' % (np.mean(predicted_class == train_labels))


# evaluate testing set accuracy
hidden_layer1 = np.maximum(0, np.dot(test_dataset, W) + b)
hidden_layer2=np.maximum(0,np.dot(hidden_layer1,W1)+b1)
scores = np.dot(hidden_layer2, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print 'testing accuracy: %.2f' % (np.mean(predicted_class == test_labels))


