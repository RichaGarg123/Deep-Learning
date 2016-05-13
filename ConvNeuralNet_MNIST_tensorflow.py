# -*- coding: utf-8 -*-
"""
Created on Thu May 12 23:07:13 2016

@author: RICHA
"""

import cPickle as pickle 
'''load MNIST data '''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

''' Start TensorFlow Interative Session'''
import tensorflow as tf
sess = tf.InteractiveSession()

''' Multilayer Convolution Neural Network'''
''' Defining methods for Weights and biases '''

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
''' Defining methods for Convolution Layer and Pooling '''
'''stride is 1 x 1'''
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
'''pool filter size is 2 x 2 and stride is 2 x2 '''
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

''' first convolution layer, filter size = 5 x 5 x 1 and number of filters =32'''
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

'''reshaping the image from 784 to 28 x28 '''
x_image = tf.reshape(x, [-1,28,28,1])

''' applying convolution on the image x.W+b and then applying ReLU and then pooling '''
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

'''second convolution layer, filter size: 5 x 5 x 32, number of filters = 64'''
'''applying convolution, reLU and then pooling'''
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

''' since we applied pooling twice 28 x 28 reduced to 14 x 14 and then 7 x 7'''
'''Now add a fully connected layer with 1024 neurons '''
'''reshape tensor and then apply perform wx+b and then ReLU '''
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

''' drop out applied to avoid overfitting '''
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

'''applying softmax layer'''
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

''' Training and evaluation of the model'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
