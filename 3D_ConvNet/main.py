'''
Author: Aman Raa
Contact: arana@wpi.edu

Topic: This file is the main file controlling batch creation and training
'''


import tensorflow as tf
import numpy as np
import cv2, os
import get_training_data as TrainingData
import model


def train():

	generator = TrainingData.batch_generator()  # Generates a batch of shape 128 x 512 x 512 x 2

	## Training
	with tf.name_scope('Placeholders'):

		x = tf.placeholder(tf.float32, shape=[None, 2, 512, 512, 1], name='x')
		y = tf.placeholder(tf.float32, name='y')


	init = tf.global_variables_initializer()

	prediction = model.Model(x)
	print '\n\n', prediction, '\n\n'


	with tf.name_scope('Loss'):

		loss = tf.losses.mean_squared_error(y, prediction, weights=406.0)
		# tf.summary.scalar('Loss', loss)


	with tf.name_scope('Optimizer'):

		optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)


	with tf.Session() as sess:

		sess.run(init)

		# merged = tf.summary.merge_all()
		# log_summary = tf.summary.FileWriter('.', sess.graph)

		counter = 0

		for batch in generator:

			counter += 1
			
			batch_x = np.array([X for X, _ in batch]).reshape([16, 2, 512, 512, 1]).astype(np.float32)
			batch_y = np.array([Y for _, Y in batch])
			batch_y = batch_y.astype(np.float32)

			# print batch_x.shape, batch_y.shape

			l, _ = sess.run([loss, optimizer], feed_dict={x: batch_x, y: batch_y})			
			# log_summary.add_summary(_summary, counter)

			print 'Iteration: {0}\tLoss: {1}'.format(counter, l)

			# if counter > 100:
			# 	break


train()