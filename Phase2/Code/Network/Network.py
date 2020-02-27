"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def CBR(inputToLayer, numFilters, kernel_size):
    CBR1 = tf.layers.conv2d(inputs = inputToLayer, padding='same',filters = numFilters, kernel_size = kernel_size, activation = None)
    CBR2 = tf.layers.batch_normalization(inputs = CBR1, axis = -1, center = True, scale = True)
    outputOfLayer = tf.nn.relu(CBR2)
    return outputOfLayer

def HomographyModel(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################
    net = Img
    net = CBR(net, 64, 3)
    #net = tf.nn.dropout(net, 0.5)
    net = CBR(net, 64, 3)

    net = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)

    net = CBR(net, 64, 3)
    net = CBR(net, 64, 3)
    net = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)

    net = CBR(net, 128, 3)
    net = CBR(net, 128, 3)
    net = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)

    net = CBR(net, 128, 3)
    net = CBR(net, 128, 3)
    #net = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)

    # net = CBR(net, 256, 3)
    # net = CBR(net, 256, 3)
    # net = CBR(net, 256, 3)
    # net = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)
    
    net = tf.layers.flatten(net)

    #Define the Neural Network's fully connected layers:
    net = tf.layers.dense(inputs = net, name ='layer_fc1', units = 1024, activation = tf.nn.relu)
    net = tf.nn.dropout(net, 0.5)
    #net = tf.layers.dense(inputs = net, name ='layer_fc2',units = 128, activation=tf.nn.relu)
    H4Pt = tf.layers.dense(inputs = net, name='layer_fc_out', units = 8, activation = None)

    #prLogits is defined as the final output of the neural network
    # prLogits = layer_fc2
    # H4Pt = net
    #prSoftMax is defined as normalized probabilities of the output of the neural network
    # prSoftMax = tf.nn.softmax(logits = prLogits)
    # print(prLogits)
    return H4Pt

