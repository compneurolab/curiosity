# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import hdf5provider

IMAGE_SIZE = 256
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 1
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.


tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
FLAGS = tf.app.flags.FLAGS


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data


def error_rate(predictions, imgs):
  """Return the error rate based on dense predictions and sparse labels."""
  return np.abs(predictions - imgs).mean()


def main(argv=None):  # pylint: disable=unused-argument
    # Get the data.

  # Extract it into np arrays.
  hdf5source = '/data/imagenet_dataset/hdf5_cached_from_om7/data.raw'
  sourcelist = ['data']
  postprocess = {'data': lambda x, _: x.reshape((x.shape[0], 3, 256, 256)).swapaxes(1, 2).swapaxes(2, 3)}
  train_slice = np.zeros(1290129).astype(np.bool); train_slice[:1000000] = True
  validation_slice = np.zeros(1290129).astype(np.bool); validation_slice[1000000: 1000512] = True
  test_slice = np.zeros(1290129).astype(np.bool); test_slice[1000512: 1001024] = True
  train_data = hdf5provider.HDF5DataProvider(hdf5source, sourcelist, BATCH_SIZE,
                                     postprocess=postprocess, subslice = train_slice)
  validation_data = hdf5provider.HDF5DataProvider(hdf5source, sourcelist, BATCH_SIZE,
                                     postprocess=postprocess, subslice = validation_slice)
  validation_data = np.row_stack([validation_data.getBatch(i)['data'] for i in range(8)])
  test_data = hdf5provider.HDF5DataProvider(hdf5source, sourcelist, BATCH_SIZE,
                                     postprocess=postprocess, subslice = test_slice)
  test_data = np.row_stack([test_data.getBatch(i)['data'] for i in range(8)]) 


  num_epochs = NUM_EPOCHS
  train_size = train_data.sizes['data'][0]

  train_data_node = tf.placeholder(
      tf.float32,
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  eval_data = tf.placeholder(
      tf.float32,
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  conv1_weights = tf.Variable(
      tf.truncated_normal([7, 7, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED),
      name = 'conv1w' )
  conv1_biases = tf.Variable(tf.zeros([32]), name='conv1b')

  conv2_weights = tf.Variable(
      tf.truncated_normal([7, 7, 32, 1],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED),
      name = 'conv2w' )
  conv2_biases = tf.Variable(tf.zeros([1]), name='conv2b')

  fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal(
          [IMAGE_SIZE // 2 * IMAGE_SIZE // 2 * 32, 256],
          stddev=0.1,
          seed=SEED), name='fc1w')
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[256]), name='fc1b')

  fc2_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal(
          [256, IMAGE_SIZE // 2 * IMAGE_SIZE // 2 * 32],
          stddev=0.1,
          seed=SEED), name='fc1w')
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[IMAGE_SIZE // 2 * IMAGE_SIZE // 2 * 32]), name='fc1b')

  def model(data, train=False):
    """The Model definition."""

    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    conv = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

    pool = tf.nn.max_pool(conv,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

    pool_shape = pool.get_shape().as_list()
    flatten = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

    encode = tf.matmul(flatten, fc1_weights) + fc1_biases

    hidden = tf.matmul(encode, fc2_weights) + fc2_biases
    
    hidden_shape = hidden.get_shape().as_list()
    unflatten = tf.reshape(hidden, [hidden_shape[0], IMAGE_SIZE//2, IMAGE_SIZE//2, 32])

    unpool = tf.image.resize_images(unflatten, IMAGE_SIZE, IMAGE_SIZE)

    conv = tf.nn.conv2d(unpool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    conv = tf.nn.bias_add(conv, conv2_biases)
    
    return conv
  
  train_prediction = model(train_data_node, True)  
  loss = tf.nn.l2_loss(tf.sub(train_prediction, train_data_node))
  loss = tf.mul(loss, 1./1000000000000)

  #regularizers = tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases)
  #loss += 5e-4 * regularizers

  batch = tf.Variable(0, trainable=False)

  learning_rate = tf.train.exponential_decay(
      .00005,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)

  optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

  eval_prediction = model(eval_data)


  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  # Create a local session to run the training.
  start_time = time.time()
  with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()
    print('Initialized!')
    # Loop through training steps.
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      batch_data = train_data.getNextBatch()['data']
      feed_dict = {train_data_node: batch_data}
      # Run the graph and fetch some of the nodes.
      _, l, lr, predictions = sess.run(
          [optimizer, loss, learning_rate, train_prediction],
          feed_dict=feed_dict)
      print(step, l)
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.6f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.6f' % error_rate(predictions, batch_data))
        print('Validation error: %.6f' % error_rate(
               eval_in_batches(validation_data, sess), validation_data))
        sys.stdout.flush()
    # Finally print the result!
    test_error = error_rate(eval_in_batches(test_data, sess), test_data)
    print('Test error: %.4f' % test_error)
    if FLAGS.self_test:
      print('test_error', test_error)
      assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
          test_error,)


if __name__ == '__main__':
  tf.app.run()
