'''
Author: Aman Rana
Contact: arana@wpi.edu

Topic: This file contains code for Decoder
'''


import tensorflow as tf
import numpy as np


with tf.device('/cpu:0'):

	with tf.name_scope('Decoder_Parameters'):

		with tf.name_scope('Weights'):

			W = {

				1: tf.Variable(tf.truncated_normal([3, 3, 256, 128])),
				2: tf.Variable(tf.truncated_normal([3, 3, 128, 128])),
				3: tf.Variable(tf.truncated_normal([3, 3, 128, 64])),
				4: tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
				5: tf.Variable(tf.truncated_normal([3, 3, 64, 32])),
				6: tf.Variable(tf.truncated_normal([3, 3, 32, 32])),
				7: tf.Variable(tf.truncated_normal([3, 3, 32, 1]))

			}

		with tf.name_scope('Biases'):

			B = {

				1: tf.Variable(tf.truncated_normal([128])),
				2: tf.Variable(tf.truncated_normal([128])),
				3: tf.Variable(tf.truncated_normal([64])),
				4: tf.Variable(tf.truncated_normal([64])),
				5: tf.Variable(tf.truncated_normal([32])),
				6: tf.Variable(tf.truncated_normal([32])),
				7: tf.Variable(tf.truncated_normal([1]))

			}


def decoder(x, k):

	with tf.name_scope('Decoder'):


		with tf.name_scope('Layer1'):

			reshaped = tf.reshape(x, [k, 4, 4, 256])


		with tf.name_scope('Layer2'):

			conv1 = tf.nn.conv2d(reshaped, W[1], strides=[1, 1, 1, 1], padding='SAME') + B[1]
			# relu1 = tf.nn.relu(conv1)

			# conv2 = tf.nn.conv2d(relu1, W[2], strides=[1, 1, 1, 1], padding='SAME') + B[2]
			# relu2 = tf.nn.relu(conv2)

			# pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			up1 = tf.image.resize_nearest_neighbor(conv1, [8, 8])


		with tf.name_scope('Layer3'):

			conv2 = tf.nn.conv2d(up1, W[2], strides=[1, 1, 1, 1], padding='SAME') + B[2]
			# relu2 = tf.nn.relu(conv2)

			# conv4 = tf.nn.conv2d(relu3, W[4], strides=[1, 1, 1, 1], padding='SAME') + B[4]
			# relu4 = tf.nn.relu(conv4)

			# pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			up2 = tf.image.resize_nearest_neighbor(conv2, [16, 16])


		with tf.name_scope('Layer4'):

			conv3 = tf.nn.conv2d(up2, W[3], strides=[1, 1, 1, 1], padding='SAME') + B[3]
			# relu3 = tf.nn.relu(conv3)

			# conv6 = tf.nn.conv2d(relu5, W[6], strides=[1, 1, 1, 1], padding='SAME') + B[6]
			# relu6 = tf.nn.relu(conv6)

			# pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			up3 = tf.image.resize_nearest_neighbor(conv3, [32, 32])


		with tf.name_scope('Layer5'):

			conv4 = tf.nn.conv2d(up3, W[4], strides=[1, 1, 1, 1], padding='SAME') + B[4]
			# relu4 = tf.nn.relu(conv4)

			# conv8 = tf.nn.conv2d(relu7, W[8], strides=[1, 1, 1, 1], padding='SAME') + B[8]
			# relu8 = tf.nn.relu(conv8)

			# pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			up4 = tf.image.resize_nearest_neighbor(conv4, [64, 64])


		with tf.name_scope('Layer6'):

			conv5 = tf.nn.conv2d(up4, W[5], strides=[1, 1, 1, 1], padding='SAME') + B[5]
			# relu5 = tf.nn.relu(conv5)

			# conv10 = tf.nn.conv2d(relu9, W[10], strides=[1, 1, 1, 1], padding='SAME') + B[10]
			# relu10 = tf.nn.relu(conv10)

			# pool5 = tf.nn.max_pool(relu5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			up5 = tf.image.resize_nearest_neighbor(conv5, [128, 128])


		with tf.name_scope('Layer7'):

			conv6 = tf.nn.conv2d(up5, W[6], strides=[1, 1, 1, 1], padding='SAME') + B[6]
			# relu6 = tf.nn.relu(conv6)

			# conv8 = tf.nn.conv2d(relu7, W[8], strides=[1, 1, 1, 1], padding='SAME') + B[8]
			# relu8 = tf.nn.relu(conv8)

			# pool6 = tf.nn.max_pool(relu6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			up6 = tf.image.resize_nearest_neighbor(conv6, [256, 256])


		with tf.name_scope('Layer8'):

			conv7 = tf.nn.conv2d(up6, W[7], strides=[1, 1, 1, 1], padding='SAME') + B[7]
			# relu7 = tf.nn.relu(conv7)

			# conv10 = tf.nn.conv2d(relu9, W[10], strides=[1, 1, 1, 1], padding='SAME') + B[10]
			# relu10 = tf.nn.relu(conv10)

			# pool7 = tf.nn.max_pool(relu7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			up7 = tf.image.resize_nearest_neighbor(conv7, [512, 512])


		return up7