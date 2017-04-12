'''
Author: Aman Rana
Contact: arana@wpi.edu

Topic: This file contains code for 3D ConvNet.
'''

import tensorflow as tf
import numpy as np
from PARAMETERS import *


class Model:

	def predict(self, x, W, B, beta1, gamma1, beta2, gamma2, beta3, gamma3):

		with tf.name_scope('Model'):
		

			with tf.name_scope('ConvLayer1'):

				conv1 = tf.nn.conv3d(x, W[1], strides=[1, 1, 1, 1, 1], padding='SAME') + B[1]
				conv1 = tf.nn.conv3d(conv1, W[2], strides=[1, 1, 1, 1, 1], padding='SAME') + B[2]
				relu1 = tf.nn.relu(conv1)

				pool1 = tf.nn.max_pool3d(relu1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
				mean1, variance1 = tf.nn.moments(pool1, axes=[0, 2, 3], keep_dims=True)
				pool1 = tf.nn.batch_normalization(pool1, mean=mean1, variance=variance1, offset=beta1, scale=gamma1, variance_epsilon=1e-5)


			with tf.name_scope('ConvLayer2'):

				conv2 = tf.nn.conv3d(pool1, W[3], strides=[1, 1, 1, 1, 1], padding='SAME') + B[3]
				conv2 = tf.nn.conv3d(conv2, W[4], strides=[1, 1, 1, 1, 1], padding='SAME') + B[4]
				relu2 = tf.nn.relu(conv2)

				pool2 = tf.nn.max_pool3d(relu2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
				mean2, variance2 = tf.nn.moments(pool2, axes=[0, 2, 3], keep_dims=True)
				pool2 = tf.nn.batch_normalization(pool2, mean=mean2, variance=variance2, offset=beta2, scale=gamma2, variance_epsilon=1e-5)


			with tf.name_scope('ConvLayer3'):

				conv3 = tf.nn.conv3d(pool2, W[5], strides=[1, 1, 1, 1, 1], padding='SAME') + B[5]
				conv3 = tf.nn.conv3d(conv3, W[6], strides=[1, 1, 1, 1, 1], padding='SAME') + B[6]
				relu3 = tf.nn.relu(conv3)

				pool3 = tf.nn.max_pool3d(relu3, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
				mean3, variance3 = tf.nn.moments(pool3, axes=[0, 2, 3], keep_dims=True)
				pool3 = tf.nn.batch_normalization(pool3, mean=mean3, variance=variance3, offset=beta3, scale=gamma3, variance_epsilon=1e-5)


			with tf.name_scope('Flatten'):

				dim = tf.reduce_prod(pool3.get_shape().as_list()[1:])
				flatten = tf.reshape(pool3, [-1, dim])


			with tf.name_scope('FullyConnected1'):

				fc1 = tf.add(tf.matmul(flatten, W[9]), B[9])
				fc1 = tf.nn.sigmoid(fc1)
				fc1 = tf.nn.dropout(fc1, keep_prob=DROPOUT_PROB)


			with tf.name_scope('FullyConnected2'):

				fc2 = tf.add(tf.matmul(fc1, W[10]), B[10])
				fc2 = tf.nn.sigmoid(fc2)
				fc2 = tf.nn.dropout(fc2, keep_prob=DROPOUT_PROB)


			with tf.name_scope('Output'):

				output = tf.add(tf.matmul(fc2, W[11]), B[11])
				output = tf.nn.softmax(output)


		return tf.clip_by_value(output, 1e-3, 1.0)
