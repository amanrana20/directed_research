'''
Author: Aman Rana
Contact: arana@wpi.edu

Topic: This file contains code for 3D ConvNet.
'''


import tensorflow as tf
import numpy as np


verbose = False


with tf.device('/cpu:0'):

	with tf.name_scope('Encoder_Parameters'):

		with tf.name_scope('Weights'):

			W = {

				1: tf.Variable(tf.truncated_normal([3, 3, 1, 32])),
				2: tf.Variable(tf.truncated_normal([3, 3, 32, 32])),
				3: tf.Variable(tf.truncated_normal([3, 3, 32, 64])),
				4: tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
				5: tf.Variable(tf.truncated_normal([3, 3, 64, 128])),
				6: tf.Variable(tf.truncated_normal([3, 3, 128, 128])),
				7: tf.Variable(tf.truncated_normal([3, 3, 128, 256])),
				8: tf.Variable(tf.truncated_normal(4096, 512)),
				9: tf.Variable(tf.truncated_normal(512, 64)),
				10: tf.Variable(tf.truncated_normal(64, 2))

			}

		with tf.name_scope('Biases'):

			B = {

				1: tf.Variable(tf.truncated_normal([32])),
				2: tf.Variable(tf.truncated_normal([32])),
				3: tf.Variable(tf.truncated_normal([64])),
				4: tf.Variable(tf.truncated_normal([64])),
				5: tf.Variable(tf.truncated_normal([128])),
				6: tf.Variable(tf.truncated_normal([128])),
				7: tf.Variable(tf.truncated_normal([256])),
				8: tf.Variable(tf.truncated_normal([512])),
				9: tf.Variable(tf.truncated_normal([64])),
				10: tf.Variable(tf.truncated_normal([2]))

			}


def encoder(x):

	with tf.name_scope('Encoder'):


		with tf.name_scope('ConvLayer1'):

			conv1 = tf.nn.conv2d(x, W[1], strides=[1, 1, 1, 1], padding='SAME') + B[1]
			relu1 = tf.nn.relu(conv1)

			# conv2 = tf.nn.conv2d(relu1, W[2], strides=[1, 1, 1, 1], padding='SAME') + B[2]
			# relu2 = tf.nn.relu(conv2)

			pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


		with tf.name_scope('ConvLayer2'):

			conv2 = tf.nn.conv2d(pool1, W[2], strides=[1, 1, 1, 1], padding='SAME') + B[2]
			relu2 = tf.nn.relu(conv2)

			# conv4 = tf.nn.conv2d(relu3, W[4], strides=[1, 1, 1, 1], padding='SAME') + B[4]
			# relu4 = tf.nn.relu(conv4)

			pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


		with tf.name_scope('ConvLayer3'):

			conv3 = tf.nn.conv2d(pool2, W[3], strides=[1, 1, 1, 1], padding='SAME') + B[3]
			relu3 = tf.nn.relu(conv3)

			# conv6 = tf.nn.conv2d(relu5, W[6], strides=[1, 1, 1, 1], padding='SAME') + B[6]
			# relu6 = tf.nn.relu(conv6)

			pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


		with tf.name_scope('ConvLayer4'):

			conv4 = tf.nn.conv2d(pool3, W[4], strides=[1, 1, 1, 1], padding='SAME') + B[4]
			relu4 = tf.nn.relu(co+nv4)

			# conv8 = tf.nn.conv2d(relu7, W[8], strides=[1, 1, 1, 1], padding='SAME') + B[8]
			# relu8 = tf.nn.relu(conv8)

			pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


		with tf.name_scope('ConvLayer5'):

			conv5 = tf.nn.conv2d(pool4, W[5], strides=[1, 1, 1, 1], padding='SAME') + B[5]
			relu5 = tf.nn.relu(conv5)

			# conv10 = tf.nn.conv2d(relu9, W[10], strides=[1, 1, 1, 1], padding='SAME') + B[10]
			# relu10 = tf.nn.relu(conv10)

			pool5 = tf.nn.max_pool(relu5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


		with tf.name_scope('ConvLayer6'):

			conv6 = tf.nn.conv2d(pool5, W[6], strides=[1, 1, 1, 1], padding='SAME') + B[6]
			relu6 = tf.nn.relu(conv6)

			# conv8 = tf.nn.conv2d(relu7, W[8], strides=[1, 1, 1, 1], padding='SAME') + B[8]
			# relu8 = tf.nn.relu(conv8)

			pool6 = tf.nn.max_pool(relu6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


		with tf.name_scope('ConvLayer7'):

			conv7 = tf.nn.conv2d(pool6, W[7], strides=[1, 1, 1, 1], padding='SAME') + B[7]
			relu7 = tf.nn.relu(conv7)

			# conv10 = tf.nn.conv2d(relu9, W[10], strides=[1, 1, 1, 1], padding='SAME') + B[10]
			# relu10 = tf.nn.relu(conv10)

			pool7 = tf.nn.max_pool(relu7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


		with tf.name_scope('Flatten'):

			flatten = tf.reshape(pool7, [1, -1])


		with tf.name_scope('FullyConnected1'):

			fc1 = tf.add(tf.matmul(flatten, W[8]), B[8])


		with tf.name_scope('FullyConnected2'):

			fc2 = tf.add(tf.matmul(fc1, W[9]), B[9])


		with tf.name_scope('Output'):

			output = tf.add(tf.matmul(fc2, W[10]), B[10])


		return output