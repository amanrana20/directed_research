'''
Author: Aman Raa
Contact: arana@wpi.edu

Topic: This file is the main file controlling batch creation and training
'''


import tensorflow as tf
import numpy as np
import cv2, os, sys
from PARAMETERS import *
from model import Model


num_positive_samples = BATCH_SIZE / 2
num_negative_samples = BATCH_SIZE / 2

## Create batch generator
def batch_generator(path_to_processed_LUNA_dataset):

	'''
	Since the ratio of positive to negative samples is very low, 
	there is a huge class imbalance. This batch generator creates 
	a batch while preserving the class ratio in the batch.

	Return: (batch)
	'''

	all_positive_samples = os.listdir(os.path.join(path_to_processed_LUNA_dataset, 'Cancer'))
	all_negative_samples = os.listdir(os.path.join(path_to_processed_LUNA_dataset, 'Non Cancer'))

	counter_positive = 0
	counter_negative = 0

	new_data = True  # This variable tells whethe there is more data or not
	
	while new_data:

		start_pos_positive = counter_positive
		start_pos_negative = counter_negative
		end_pos_positive = start_pos_positive + num_positive_samples
		end_pos_negtive = start_pos_negative + num_negative_samples

		positive_sample_names = np.array( all_positive_samples[start_pos_positive : end_pos_positive ] )
		negative_sample_names = np.array( all_negative_samples[start_pos_negative : end_pos_negtive ] )

		if not len(positive_sample_names) == num_positive_samples:
			break

		batch_sample_names = np.array( np.concatenate( [positive_sample_names,	negative_sample_names] ) )

		counter_positive = end_pos_positive
		counter_negative = end_pos_negtive
		
		# Creating numpy array to hold batch
		batch = []

		for i, each_sample_name in enumerate(batch_sample_names):

			folder = 'Cancer' if i < num_positive_samples else 'Non Cancer'
			
			path_sample = os.path.join(path_to_processed_LUNA_dataset, folder, each_sample_name)
			sample_data = np.load(path_sample)

			label = np.array([1, 0]) if folder == 'Cancer' else np.array([0, 1])

			batch.append([sample_data, label])

		np.random.shuffle(batch)
		batch = np.array(batch)
		print batch.shape

		if not batch.shape == (BATCH_SIZE, 2):
			new_data = False
			break
		else:
			yield batch


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
				8: tf.Variable(tf.random_normal([3, 3, 3, 256, 256], stddev=0.5)),
				9: tf.Variable(tf.random_normal([16384, 1024], stddev=0.5)),
				10: tf.Variable(tf.random_normal([1024, 256], stddev=0.5)),
				11: tf.Variable(tf.random_normal([256, 2], stddev=0.5))

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
				8: tf.Variable(tf.random_normal([256])),
				9: tf.Variable(tf.random_normal([1024])),
				10: tf.Variable(tf.random_normal([256])),
				11: tf.Variable(tf.random_normal([2]))

			}

		beta1 = tf.Variable(0.0, [5, 32, 32, 32])
		gamma1 = tf.Variable(1.0, [5, 32, 32, 32])
		beta2 = tf.Variable(0.0, [3, 16, 16, 64])
		gamma2 = tf.Variable(1.0, [3, 16, 16, 64])
		beta3 = tf.Variable(0.0, [2, 8, 8, 128])
		gamma3 = tf.Variable(1.0, [2, 8, 8, 128])




def train_tf(path_to_processed_LUNA_dataset, path_to_folder_to_save_checkpoints):
	## Training
	with tf.name_scope('Placeholders'):
		x = tf.placeholder(tf.float32, shape=[None, NUM_SLICES, IMAGE_SIZE, IMAGE_SIZE, 1], name='X')
		y = tf.placeholder(tf.float32, name='Y')

	prediction = Model().predict(x, W, B, beta1, gamma1, beta2, gamma2, beta3, gamma3)
	
	with tf.name_scope('Loss'):
		reg_losses = tf.nn.l2_loss(W[1]) + tf.nn.l2_loss(W[2]) + tf.nn.l2_loss(W[3]) + tf.nn.l2_loss(W[4]) + \
					 tf.nn.l2_loss(W[5]) + tf.nn.l2_loss(W[6]) + tf.nn.l2_loss(W[7]) + tf.nn.l2_loss(W[8]) + \
					 tf.nn.l2_loss(W[9]) + tf.nn.l2_loss(W[10]) + tf.nn.l2_loss(W[11])
		cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)) + (1e-5 * reg_losses)
		tf.summary.scalar('Loss', cost)

	with tf.name_scope('Optimizer'):
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  		with tf.control_dependencies(update_ops):
			optimizer = tf.train.AdamOptimizer().minimize(loss=cost)

	saver = tf.train.Saver(tf.global_variables())

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		# saver.restore(sess, '/home/amanrana/Desktop/AmanRana/Checkpoints/LUNA/Epoch_10/Model_checkpoint.ckpt-45600')

		merged = tf.summary.merge_all()
		log_summary = tf.summary.FileWriter('.', sess.graph)

		counter = 0

		for epoch in range(NB_EPOCHS):
			generator = batch_generator(path_to_processed_LUNA_dataset)

			loss_history = []

			for count, batch in enumerate(generator):
				counter += 1
				try:
					batch_x = np.array([X for X, _ in batch]).astype(np.float32)
					batch_x = batch_x.reshape([BATCH_SIZE, NUM_SLICES, IMAGE_SIZE, IMAGE_SIZE, 1])
					batch_y = np.array([Y for _, Y in batch]).reshape([BATCH_SIZE, 2])
					batch_y = batch_y.astype(np.float32)

					_summary, p, l, _ = sess.run([merged, prediction, cost, optimizer], feed_dict={x: batch_x, y: batch_y})			
					log_summary.add_summary(_summary, counter)
					loss_history.append(float(l))
					print p
					print batch_y

					print 'Iteration: {0}\tLoss: {1}'.format(counter, l)

					if len(batch) != BATCH_SIZE:
						break
				except:
					continue

			os.system('mkdir {}/Epoch_{}'.format(path_to_folder_to_save_checkpoints, epoch+1))
			save_path = saver.save(sess, '{}/Epoch_{}/Model_checkpoint.ckpt'.format(path_to_folder_to_save_checkpoints, epoch+1), global_step=counter)
			print('Saved model to {}'.format(save_path))



if __name__ == '__main__':
	args = sys.argv
	train_tf(args[1], args[2])