# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:16:37 2019

@author: sb00747428
"""

import tensorflow.contrib.layers as lays
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


train_data = np.load('train_data.npy')
train_labels= np.load('train_labels.npy')
batch_size = 40  # Number of samples in each batch
epoch_num = 500     # Number of epochs to train the network
lr = 0.001
BATCH_SIZE = 40
TRAIN_BUF = 4400

#train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
def autoencoder(inputs):
    # encoder

    net = lays.conv2d(inputs, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 8, [5, 5], stride=2, padding='SAME')
    # decoder

    net = lays.conv2d_transpose(net, 16, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, 3, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.relu)
    return net

ae_inputs = tf.placeholder(tf.float32, (None, 40, 32, 3))  # input to the network (MNIST images)
ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network
# calculate the loss and optimize the network
loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # claculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
# initialize the network
init = tf.global_variables_initializer()

       # Learning rate

y = train_data.shape[0]

# calculate the number of batches per epoch
batch_per_ep = train_data.shape[0] // batch_size
with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch_num):  # epochs loop
        for i in range (0,y,batch_size):  # batches loop
            batch_img = train_data[i:i+40,:,:,:]              # reshape each sample to an (28, 28) image
        #batch_img = resize_batch(batch_img)                          # reshape the images to (32, 32)
            _, c = sess.run([train_op, loss], feed_dict={ae_inputs: batch_img})
        print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))
    # test the trained network

    recon_img = sess.run([ae_outputs], feed_dict={ae_inputs: batch_img})[0]
    # plot the reconstructed images and their ground truths (inputs)
    plt.figure(1)
    plt.title('Reconstructed Images')
    for i in range(40):
        plt.subplot(4, 10, i+1)
        plt.imshow(recon_img[i])
    plt.figure(2)
    plt.title('Input Images')
    for i in range(40):
        plt.subplot(4, 10, i+1)
        plt.imshow(batch_img[i])
    plt.show()
