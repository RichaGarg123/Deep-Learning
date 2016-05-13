# -*- coding: utf-8 -*-
"""
Created on Sun May 08 06:47:48 2016

@author: RICHA
"""

import numpy as np
import struct

import os
print os.getcwd()




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
h = 100 # size of hidden layer
W = np.random.uniform(low=-np.sqrt(6./(D+h)),high=np.sqrt(6./(D+h)),size=(D,h))
b = np.zeros((1,h))
W2 = np.random.uniform(low=-np.sqrt(6./(K+h)),high=np.sqrt(6./(K+h)),size=(h,K))
b2 = np.zeros((1,K))

step_size = 1e-01
reg = 1e-3 
v1=0
v2=0
v3=0
v4=0
mu=0.9
num_examples = train_dataset.shape[0]
for i in xrange(10000):
  
  # evaluate class scores, [N x K]
  hidden_layer = np.maximum(0, np.dot(train_dataset, W) + b) # note, ReLU activation
  scores = np.dot(hidden_layer, W2) + b2
  
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  
  
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples),train_labels])
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
  loss = data_loss + reg_loss
  if i % 10 == 0:
    print "iteration %d: loss %f" % (i, loss)
  
  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),train_labels] -= 1
  dscores /= num_examples
  
  # backpropate the gradient to the parameters
  # first backprop into parameters W2 and b2
  dW2 = np.dot(hidden_layer.T, dscores)
  db2 = np.sum(dscores, axis=0, keepdims=True)
  # next backprop into hidden layer
  dhidden = np.dot(dscores, W2.T)
  # backprop the ReLU non-linearity
  dhidden[hidden_layer <= 0] = 0
  # finally into W,b
  dW = np.dot(train_dataset.T, dhidden)
  db = np.sum(dhidden, axis=0, keepdims=True)
  
  # add regularization gradient contribution
  
  dW2 += reg * W2
  dW += reg * W
  
  

  # perform a parameter update  with momentum update
  v1=mu*v1 - step_size *dW
  W +=v1
  v2=mu*v2 - step_size *db
  b +=v2
  v3=mu*v3- step_size* dW2
  W2 +=v3
  v4=mu*v4 - step_size* db2
  b2 +=v4



# evaluate training set accuracy
hidden_layer = np.maximum(0, np.dot(train_dataset, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print 'training accuracy: %.2f' % (np.mean(predicted_class == train_labels))


# evaluate testing set accuracy
hidden_layer = np.maximum(0, np.dot(test_dataset, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print 'testing accuracy: %.2f' % (np.mean(predicted_class == test_labels))


