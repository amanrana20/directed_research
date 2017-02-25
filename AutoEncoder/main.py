'''
Author: Aman Rana
email: arana@wpi.edu

About: This sile contains code for overall code management and execution
'''


import numpy as np
import cv2, os
import SimpleITK as itk
import tensorflow as tf
from Encoder import *
from Decoder import *


# Costants
PATH_TRAINING_DATASET = '../../../kaggle_main/Data Science Bowl Kaggle/dataset/Annotated Lung Cancer Dataset/training data'
EPOCHS = 5
GENERATOR_BATCH_SIZE = 8



training_files = os.listdir(PATH_TRAINING_DATASET)
training_files.sort()
l = len(training_files)


## generating batch
def get_batch():
    
	for i in range(0, l, GENERATOR_BATCH_SIZE*2):

		batch_files = training_files[i: i+GENERATOR_BATCH_SIZE*2]

		l1 = len(batch_files)

		X = np.zeros([l1/2, 512, 512, 1])
		Y = np.zeros([l1/2, 512, 512, 1])

		for j in range(0, l1, 2):

		    x = batch_files[j]
		    y = batch_files[j+1]

		    path_to_x = os.path.join(PATH_TRAINING_DATASET, x)
		    path_to_y = os.path.join(PATH_TRAINING_DATASET, y)

		    img = np.array(np.load(path_to_x))[1, :, :]
		    mask = np.array(np.load(path_to_y))[1, :, :]

		    X[j/2, :, :, :] = img.reshape([512, 512, 1])
		    Y[j/2, :, :, :] = mask.reshape([512, 512, 1])

		yield {'X': X, 'Y': Y}


def loss_calc(y_true, y_pred, k):

	smooth = 1

	intersection_arr = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[1, 2])

	numerator = (2 * intersection_arr + smooth)
	denominator = (tf.reduce_sum(y_true, axis=[1,2]) + tf.reduce_sum(y_pred, axis=[1,2]) + smooth)

	dice_coeffs = numerator / denominator
	avg_dice_coeff = tf.reduce_sum(dice_coeffs)

	return avg_dice_coeff


def calc_loss(y_true, y_pred, k):

    return -loss_calc(y_true, y_pred, k)


## train
def train():

	with tf.name_scope('Placeholders'):

		x = tf.placeholder(tf.float32, shape=[None, 512, 512, 1], name='x')
		y = tf.placeholder(tf.float32, shape=[None, 512, 512, 1], name='y')
		# k = tf.placeholder(tf.float32, name='k')

	init = tf.global_variables_initializer()

	k = GENERATOR_BATCH_SIZE
	
	prediction = decoder(encoder(x), k)


	with tf.name_scope('Loss'):

		loss = calc_loss(y, prediction, k)
		tf.summary.scalar('loss', loss)


	with tf.name_scope('Optimizer'):

		optimizer = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)


	with tf.Session() as sess:

		merged = tf.summary.merge_all()
		log_summary = tf.summary.FileWriter('.', sess.graph)

		sess.run(init)

		counter = 0

		for batch in get_batch():

			counter += 1

			batch_X = np.array(batch['X']).astype('float')
			batch_Y = np.array(batch['Y']).astype('float')

			# print batch_X.shape, batch_Y.shape

			_summary, l, _ = sess.run([merged, loss, optimizer], feed_dict={x: batch_X, y: batch_X})
			
			log_summary.add_summary(_summary, counter)

			print 'Iteration: {0}\tLoss: {1}'.format(counter, l)

			if counter >= 145:

				break

		# correct_pred = dice_coef(pred(x), y, batch_X.shape[0])
		# accuracy = tf.
		


train()
