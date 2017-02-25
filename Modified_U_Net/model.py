'''
Author: Aman Rana
email: arana@wpi.edu
'''


import tensorflow as tf
import numpy as np


def dice_coef(y_true, y_pred, k):

	smooth = 1

	intersection_arr = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[1, 2])

	numerator = (2 * intersection_arr + smooth)
	denominator = (tf.reduce_sum(y_true, axis=[1,2]) + tf.reduce_sum(y_pred, axis=[1,2]) + smooth)

	dice_coeffs = numerator / denominator
	avg_dice_coeff = tf.reduce_sum(dice_coeffs)

	return avg_dice_coeff


def dice_coef_loss(y_true, y_pred, k):

    return 1.-dice_coef(y_true, y_pred, k)



with tf.device('/cpu:0'):

	with tf.name_scope('Parameters'):

		with tf.name_scope('Weights'):

			W = {

				1: tf.Variable(tf.truncated_normal([3, 3, 1, 32])),
				2: tf.Variable(tf.truncated_normal([3, 3, 32, 32])),
				3: tf.Variable(tf.truncated_normal([3, 3, 32, 64])),
				4: tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
				5: tf.Variable(tf.truncated_normal([3, 3, 64, 128])),
				6: tf.Variable(tf.truncated_normal([3, 3, 128, 128])),
				7: tf.Variable(tf.truncated_normal([3, 3, 128, 256])),
				8: tf.Variable(tf.truncated_normal([3, 3, 256, 256])),
				9: tf.Variable(tf.truncated_normal([3, 3, 256, 512])),
				10: tf.Variable(tf.truncated_normal([3, 3, 512, 512])),
				11: tf.Variable(tf.truncated_normal([3, 3, 768, 256])),
				12: tf.Variable(tf.truncated_normal([3, 3, 256, 256])),
				13: tf.Variable(tf.truncated_normal([3, 3, 384, 128])),
				14: tf.Variable(tf.truncated_normal([3, 3, 128, 128])),
				15: tf.Variable(tf.truncated_normal([3, 3, 192, 64])),
				16: tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
				17: tf.Variable(tf.truncated_normal([3, 3, 96, 32])),
				18: tf.Variable(tf.truncated_normal([3, 3, 32, 32])),
				19: tf.Variable(tf.truncated_normal([1, 1, 32, 1]))

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
				8: tf.Variable(tf.truncated_normal([256])),
				9: tf.Variable(tf.truncated_normal([512])),
				10: tf.Variable(tf.truncated_normal([512])),
				11: tf.Variable(tf.truncated_normal([256])),
				12: tf.Variable(tf.truncated_normal([256])),
				13: tf.Variable(tf.truncated_normal([128])),
				14: tf.Variable(tf.truncated_normal([128])),
				15: tf.Variable(tf.truncated_normal([64])),
				16: tf.Variable(tf.truncated_normal([64])),
				17: tf.Variable(tf.truncated_normal([32])),
				18: tf.Variable(tf.truncated_normal([32])),
				19: tf.Variable(tf.truncated_normal([1]))

			}


def pred(x):
	with tf.name_scope('Model'):

		with tf.name_scope('Layer1'):

			conv1 = tf.nn.conv2d(x, W[1], strides=[1, 1, 1, 1], padding='SAME') + B[1]
			relu1 = tf.nn.relu(conv1)

			conv2 = tf.nn.conv2d(relu1, W[2], strides=[1, 1, 1, 1], padding='SAME') + B[2]
			relu2 = tf.nn.relu(conv2)

			pool1 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


		with tf.name_scope('Layer2'):

			conv3 = tf.nn.conv2d(pool1, W[3], strides=[1, 1, 1, 1], padding='SAME') + B[3]
			relu3 = tf.nn.relu(conv3)

			conv4 = tf.nn.conv2d(relu3, W[4], strides=[1, 1, 1, 1], padding='SAME') + B[4]
			relu4 = tf.nn.relu(conv4)

			pool2 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


		with tf.name_scope('Layer3'):

			conv5 = tf.nn.conv2d(pool2, W[5], strides=[1, 1, 1, 1], padding='SAME') + B[5]
			relu5 = tf.nn.relu(conv5)

			conv6 = tf.nn.conv2d(relu5, W[6], strides=[1, 1, 1, 1], padding='SAME') + B[6]
			relu6 = tf.nn.relu(conv6)

			pool3 = tf.nn.max_pool(relu6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


		with tf.name_scope('Layer4'):

			conv7 = tf.nn.conv2d(pool3, W[7], strides=[1, 1, 1, 1], padding='SAME') + B[7]
			relu7 = tf.nn.relu(conv7)

			conv8 = tf.nn.conv2d(relu7, W[8], strides=[1, 1, 1, 1], padding='SAME') + B[8]
			relu8 = tf.nn.relu(conv8)

			pool4 = tf.nn.max_pool(relu8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


		with tf.name_scope('Layer5'):

			conv9 = tf.nn.conv2d(pool4, W[9], strides=[1, 1, 1, 1], padding='SAME') + B[9]
			relu9 = tf.nn.relu(conv9)

			conv10 = tf.nn.conv2d(relu9, W[10], strides=[1, 1, 1, 1], padding='SAME') + B[10]
			relu10 = tf.nn.relu(conv10)


		with tf.name_scope('Layer6'):

			upsampling1 = tf.image.resize_nearest_neighbor(relu10, [64, 64])
			upsampling1 = tf.concat([upsampling1, conv8], axis=3)

			conv11 = tf.nn.conv2d(upsampling1, W[11], strides=[1, 1, 1, 1], padding='SAME') + B[11]
			relu11 = tf.nn.relu(conv11)

			conv12 = tf.nn.conv2d(relu11, W[12], strides=[1, 1, 1, 1], padding='SAME') + B[12]
			relu12 = tf.nn.relu(conv12)


		with tf.name_scope('Layer7'):

			upsampling2 = tf.image.resize_nearest_neighbor(relu12, [128, 128])
			upsampling2 = tf.concat([upsampling2, conv6], axis=3)

			conv13 = tf.nn.conv2d(upsampling2, W[13], strides=[1, 1, 1, 1], padding='SAME') + B[13]
			relu13 = tf.nn.relu(conv13)

			conv14 = tf.nn.conv2d(relu13, W[14], strides=[1, 1, 1, 1], padding='SAME') + B[14]
			relu14 = tf.nn.relu(conv14)


		with tf.name_scope('Layer8'):

			upsampling3 = tf.image.resize_nearest_neighbor(relu14, [256, 256])
			upsampling3 = tf.concat([upsampling3, conv4], axis=3)

			conv15 = tf.nn.conv2d(upsampling3, W[15], strides=[1, 1, 1, 1], padding='SAME') + B[15]
			relu15 = tf.nn.relu(conv15)

			conv16 = tf.nn.conv2d(relu15, W[16], strides=[1, 1, 1, 1], padding='SAME') + B[16]
			relu16 = tf.nn.relu(conv16)


		with tf.name_scope('Layer9'):

			upsampling4 = tf.image.resize_nearest_neighbor(relu16, [512, 512])
			upsampling4 = tf.concat([upsampling4, conv2], axis=3)

			conv17 = tf.nn.conv2d(upsampling4, W[17], strides=[1, 1, 1, 1], padding='SAME') + B[17]
			relu17 = tf.nn.relu(conv17)

			conv18 = tf.nn.conv2d(relu17, W[18], strides=[1, 1, 1, 1], padding='SAME') + B[18]
			relu18 = tf.nn.relu(conv18)


		with tf.name_scope('Layer10'):

			conv19 = tf.nn.conv2d(relu18, W[19], strides=[1, 1, 1, 1], padding='SAME') + B[19]
			relu19 = tf.nn.relu(conv19)

		return relu19
	
