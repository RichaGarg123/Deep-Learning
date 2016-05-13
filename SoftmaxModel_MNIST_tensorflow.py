# -*- coding: utf-8 -*-
"""
Created on Thu May 12 22:03:42 2016

@author: RICHA
"""
'''load MNIST data '''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

''' Start TensorFlow Interative Session'''
import tensorflow as tf
sess = tf.InteractiveSession()

''' Softmax Regression Model'''
''' 1. Build a placeholder for MNIST Input Images and its corresponding  class labels - 1 for actual and rest will be 0'''
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

''' Now defining variables W and b for the model'''
'''Initialize W as a 784 x 10 matrix and b as a 1x 10 matrix with all zeros'''
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
'''this will assign the variables W and b with the values i.e. zeros and x and y- with input images and labels'''
sess.run(tf.initialize_all_variables())

'''now on all images x, we compute wx+b'''
y = tf.nn.softmax(tf.matmul(x,W) + b)

''' Cost function -Cross Entropy'''
'''cross entropy= -summation of actual * ln(predicted) over all values in a row - i.e. for a given image '''
''' over here we compute values for all the input images and then take the mean value'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

''' training the model using Gradient descent with step size as 0.5 '''
''' objective is to minimize the cross entropy'''
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

''' now running this optimizer repeatedly taking 50 images at a time '''
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

''' now we check if the predicted label is correct or not, '''
'''this is done by checking if the predicted label is same as actual label and create a vector of T/F or 0/1 '''
'''And we get the percentage predicted correctly '''
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
''' adds up all values of correct_prediction and takes mean'''
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
'''testing on Test data'''
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

