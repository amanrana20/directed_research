'''
Author: Aman Raa
Contact: arana@wpi.edu

Topic: This file is the main file controlling batch creation and training
'''


import tensorflow as tf
import numpy as np
import cv2, os
import get_training_data as TrainingData


## Parameters
NB_EPOCHS = 100

with tf.device('/cpu:0'):

	with tf.name_scope('Parameters'):

		with tf.name_scope('Weights'):

			W = {

				1: tf.Variable(tf.random_normal([3, 3, 3, 1, 32], stddev=0.5)),
				2: tf.Variable(tf.random_normal([3, 3, 3, 32, 32], stddev=0.5)),
				3: tf.Variable(tf.random_normal([3, 3, 3, 32, 64], stddev=0.5)),
				4: tf.Variable(tf.random_normal([3, 3, 3, 64, 64], stddev=0.5)),
				5: tf.Variable(tf.random_normal([3, 3, 3, 64, 128], stddev=0.5)),
				6: tf.Variable(tf.random_normal([3, 3, 3, 128, 128], stddev=0.5)),
				7: tf.Variable(tf.random_normal([3, 3, 3, 128, 256], stddev=0.5)),
				8: tf.Variable(tf.random_normal([768, 1024], stddev=0.5)),
				9: tf.Variable(tf.random_normal([1024, 256], stddev=0.5)),
				10: tf.Variable(tf.random_normal([256, 2], stddev=0.5))

			}

		with tf.name_scope('Biases'):

			B = {

				1: tf.Variable(tf.random_normal([32])),
				2: tf.Variable(tf.random_normal([32])),
				3: tf.Variable(tf.random_normal([64])),
				4: tf.Variable(tf.random_normal([64])),
				5: tf.Variable(tf.random_normal([128])),
				6: tf.Variable(tf.random_normal([128])),
				7: tf.Variable(tf.random_normal([256])),
				8: tf.Variable(tf.random_normal([1024])),
				9: tf.Variable(tf.random_normal([256])),
				10: tf.Variable(tf.random_normal([2]))

			}


def predict(x):

	with tf.name_scope('Model'):


		with tf.name_scope('ConvLayer1'):

			conv1 = tf.nn.conv3d(x, W[1], strides=[1, 1, 1, 1, 1], padding='SAME') + B[1]
			relu1 = tf.nn.sigmoid(conv1)

			pool1 = tf.nn.max_pool3d(relu1, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='SAME')


		with tf.name_scope('ConvLayer2'):

			conv2 = tf.nn.conv3d(pool1, W[2], strides=[1, 1, 1, 1, 1], padding='SAME') + B[2]
			relu2 = tf.nn.sigmoid(conv2)

			pool2 = tf.nn.max_pool3d(relu2, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='SAME')


		with tf.name_scope('ConvLayer3'):

			conv3 = tf.nn.conv3d(pool2, W[3], strides=[1, 1, 1, 1, 1], padding='SAME') + B[3]
			relu3 = tf.nn.sigmoid(conv3)

			pool3 = tf.nn.max_pool3d(relu3, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='SAME')


		with tf.name_scope('ConvLayer4'):

			conv4 = tf.nn.conv3d(pool3, W[4], strides=[1, 1, 1, 1, 1], padding='SAME') + B[4]
			relu4 = tf.nn.sigmoid(conv4)

			pool4 = tf.nn.max_pool3d(relu4, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='SAME')


		with tf.name_scope('ConvLayer5'):

			conv5 = tf.nn.conv3d(pool4, W[5], strides=[1, 1, 1, 1, 1], padding='SAME') + B[5]
			relu5 = tf.nn.sigmoid(conv5)

			pool5 = tf.nn.max_pool3d(relu5, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='SAME')


		with tf.name_scope('ConvLayer6'):

			conv6 = tf.nn.conv3d(pool5, W[6], strides=[1, 1, 1, 1, 1], padding='SAME') + B[6]
			relu6 = tf.nn.sigmoid(conv6)

			pool6 = tf.nn.max_pool3d(relu6, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='SAME')


		with tf.name_scope('ConvLayer7'):

			conv7 = tf.nn.conv3d(pool6, W[7], strides=[1, 1, 1, 1, 1], padding='SAME') + B[7]
			relu7 = tf.nn.sigmoid(conv7)

			pool7 = tf.nn.max_pool3d(relu7, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='SAME')


		with tf.name_scope('Flatten'):

			dim = tf.reduce_prod(pool7.get_shape().as_list()[1:])
			flatten = tf.reshape(pool7, [-1, dim])


		with tf.name_scope('FullyConnected1'):

			fc1 = tf.add(tf.matmul(flatten, W[8]), B[8])
			fc1 = tf.nn.sigmoid(fc1)


		with tf.name_scope('FullyConnected2'):

			fc2 = tf.add(tf.matmul(fc1, W[9]), B[9])
			fc2 = tf.nn.sigmoid(fc2)


		with tf.name_scope('Output'):

			output = tf.add(tf.matmul(fc2, W[10]), B[10])
			output = tf.nn.softmax(output)


	return output


def train():

	## Training
	with tf.name_scope('Placeholders'):

		x = tf.placeholder(tf.float32, shape=[None, 3, 128, 128, 1], name='x')
		y = tf.placeholder(tf.float32, name='y')


	init = tf.global_variables_initializer()
	prediction = predict(x)


	with tf.name_scope('Loss'):

		loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
		tf.summary.scalar('Loss', loss)


	with tf.name_scope('Optimizer'):

		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss=loss)


	saver = tf.train.Saver()


	with tf.Session() as sess:

		sess.run(init)

		merged = tf.summary.merge_all()
		log_summary = tf.summary.FileWriter('.', sess.graph)

		counter = 0

		for epoch in range(NB_EPOCHS):

			generator = TrainingData.batch_generator()  # Generates a batch of shape 128 x 512 x 512 x 2

			loss_history = []

			for count, batch in enumerate(generator):

				counter += 1
				
				batch_x = np.array([X for X, _ in batch]).astype(np.float32)
				batch_x = batch_x.reshape([32, 3, 128, 128, 1])
				batch_y = np.array([Y for _, Y in batch])
				batch_y = batch_y.astype(np.float32)

				# print batch_x.shape, batch_y.shape

				_summary, p, l, _ = sess.run([merged, prediction, loss, optimizer], feed_dict={x: batch_x, y: batch_y})			
				log_summary.add_summary(_summary, counter)
				loss_history.append(float(l))
				# print p

				print 'Iteration: {0}\tLoss: {1}'.format(counter, l)

				if len(batch) != 32:
					break

			os.system('mkdir Checkpoints/Epoch_{}_LossHistory_{}'.format(epoch+1, np.mean(loss_history)))
			save_path = saver.save(sess, 'Checkpoints/Epoch_{}_LossHistory_{}/Model_checkpoint.ckpt'.format(epoch+1, np.mean(loss_history)), global_step=counter)
			print('Saved model to {}'.format(save_path))


train()