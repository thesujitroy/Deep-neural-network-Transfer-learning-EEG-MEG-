# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:12:33 2019

@author: sb00747428
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from read_data import get_data
from signalProcessing import run_sig_processing
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

"""Collect all the data"""
def arrange_data(data, labels):
    output_data = list()
    output_labels = list()
    for idx in range(len(data)):
        for segment in data[idx]:
            output_data.append(np.expand_dims(segment, axis=2))
            if labels[idx][0] == 1:
                output_labels.append(0)
            else:
                output_labels.append(1)
                
    output_data = np.array(output_data)
    output_labels = np.array(output_labels)
    return output_data, output_labels

def run_classification(data, labels, session1=(1, 2, 3), session2=(4,5)):
    
    
    for subject in data:
        # if subject == 'subject1': continue
        
        input_data = list()
        target_labels = list()
        x_test = list()
        y_test = list()
        # combine trials data of target session
        [input_data.extend(data[subject]["session" + str(idx)]['input data']) for idx in session1]
        [target_labels.extend(labels[subject]["session" + str(idx)]) for idx in session1]
        input_data = np.array(input_data)
        target_labels = np.array(target_labels)


        [x_test.extend(data[subject]["session" + str(idx)]['input data']) for idx in session2]
        [y_test.extend(labels[subject]["session" + str(idx)]) for idx in session2]
        test_data = np.array(x_test)
        test_labels = np.array(y_test)
       
            
        train_data, train_labels = arrange_data(input_data, target_labels)
        test_data, test_labels = arrange_data(x_test, y_test)
        train_data = train_data.reshape(train_data.shape[0],train_data.shape[1],train_data.shape[2],train_data.shape[4])
        test_data = test_data.reshape(test_data.shape[0],test_data.shape[1],test_data.shape[2],test_data.shape[4])
        size_y, size_x = train_data[0].shape[0:2]
        return train_data, test_data, size_y, size_x, test_labels, train_labels
    
"""-----------------"""    

'''downsample'''
def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result
'''------------------------------------------------------'''

'''Upsample'''
def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result
"""-----------------"""  

'''Building Generator'''
def Generator():
  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  concat = tf.keras.layers.Concatenate()

  inputs = tf.keras.layers.Input(shape=[None,None,3])
  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concat([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

'''--------------------------------------'''
'''Building discriminator'''
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

'''---------------'''

'''Define loss'''
def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss

'''-----------'''

@tf.function
def train_step(input_image, target):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
  
def generate_images(model, test_input, tar):
  # the training=True is intentional here since
  # we want the batch statistics while running the model
  # on the test dataset. If we use training=False, we will get
  # the accumulated statistics learned from the training dataset
  # (which we don't want)
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15,15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show() 



def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()

    # Train
    for input_image, target in train_ds:
      train_step(input_image, target)

    clear_output(wait=True)
    # Test on the same image so that the progress of the model can be 
    # easily seen.
    for example_input, example_target in test_ds.take(1):
      generate_images(generator, example_input, example_target)

    

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
    
if __name__ == '__main__':
    # '''
    OUTPUT_CHANNELS = 3
    BUFFER_SIZE = 400
    BATCH_SIZE = 1
    IMG_WIDTH = 32
    IMG_HEIGHT = 40
    EPOCHS = 150
    data_src = r"Y:\Sujit Roy\data1"
    labels_src = r"Y:\Sujit Roy\labels1"
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    LAMBDA = 100
    #
    # band_type: 0: band pass feature, 1: AR PSD feature, 2: extend band
    '''load and prepare data'''
    data, labels = run_sig_processing(data_src, labels_src, band_type=2)

    train_data, test_data, size_y, size_x, test_labels, train_labels = run_classification(data, labels)
    '''-----------------------------'''
    '''playing with generator'''
    generator = Generator()

    gen_output = generator(inp[tf.newaxis,...], training=False)
    plt.imshow(gen_output[0,...])
    '''-----------------------------'''
    
    '''playing with discriminator'''
    
    discriminator = Discriminator()
    disc_out = discriminator([inp[tf.newaxis,...], gen_output], training=False)
    plt.imshow(disc_out[0,...,-1], vmin=-20, vmax=20, cmap='RdBu_r')
    plt.colorbar()
    
    '''----------------------------'''
    
    fit(train_data, EPOCHS, test_data)