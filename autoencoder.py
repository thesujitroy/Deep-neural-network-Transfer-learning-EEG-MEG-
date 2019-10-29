# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:16:37 2019

@author: sb00747428
"""

import tensorflow.contrib.layers as lays
import numpy as np
import tensorflow as tf

train_data = np.load('train_data.npy')
train_labels= np.load('train_labels.npy')

def autoencoder(inputs):
    # encoder
    # 32 x 32 x 1   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  8 x 8 x 16
    # 8 x 8 x 16    ->  2 x 2 x 8
    net = lays.conv2d(inputs, [40,32], [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, [20, 16], [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, [10,8], [5, 5], stride=4, padding='SAME')
    # decoder
    # 2 x 2 x 8    ->  8 x 8 x 16
    # 8 x 8 x 16   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  32 x 32 x 1
    net = lays.conv2d_transpose(net, [20,16], [5, 5], stride=4, padding='SAME')
    net = lays.conv2d_transpose(net, [40,32], [5, 5], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
    return net

ae_inputs = tf.placeholder(tf.float32, (None, 40, 32, 1))  # input to the network (MNIST images)
ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network
# calculate the loss and optimize the network
loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # claculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
# initialize the network
init = tf.global_variables_initializer()

batch_size = 40  # Number of samples in each batch
epoch_num = 5     # Number of epochs to train the network
lr = 0.001        # Learning rate
# read MNIST dataset

# calculate the number of batches per epoch
batch_per_ep = train_data.shape[0] // batch_size
with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch_num):  # epochs loop
        for batch_n in range(batch_per_ep):  # batches loop
            batch_img, batch_label = train_data.next_batch(batch_size)  # read a batch
            batch_img = batch_img.reshape((-1, 40, 32, 1))               # reshape each sample to an (28, 28) image
            #batch_img = resize_batch(batch_img)                          # reshape the images to (32, 32)
            _, c = sess.run([train_op, loss], feed_dict={ae_inputs: batch_img})
            print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))
    # test the trained network
    batch_img, batch_label = train_data.next_batch(50)
    batch_img = resize_batch(batch_img)
    recon_img = sess.run([ae_outputs], feed_dict={ae_inputs: batch_img})[0]
    # plot the reconstructed images and their ground truths (inputs)
    plt.figure(1)
    plt.title('Reconstructed Images')
    for i in range(50):
        plt.subplot(5, 10, i+1)
        plt.imshow(recon_img[i, ..., 0], cmap='gray')
    plt.figure(2)
    plt.title('Input Images')
    for i in range(50):
        plt.subplot(5, 10, i+1)
        plt.imshow(batch_img[i, ..., 0], cmap='gray')
    plt.show()