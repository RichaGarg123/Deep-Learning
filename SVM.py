# -*- coding: utf-8 -*-
"""
Created on Thu May 12 21:45:51 2016

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

train_data = read_image_file('D:\\\\train-images.idx3-ubyte')

test_data=read_image_file('D:\\t10k-images.idx3-ubyte')

train_label=read_label_file('D:\\train-labels.idx1-ubyte')
test_label=read_label_file('D:\\t10k-labels.idx1-ubyte')


from sklearn.svm import SVC
poly_svm = SVC(kernel='poly',gamma='auto')
poly_svm.fit(train_data,train_label)
pred_poly = poly_svm.predict(test_data)

np.sum(pred_poly==test_label)/10000.0