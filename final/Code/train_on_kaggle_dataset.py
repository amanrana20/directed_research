import tensorflow as tf
import numpy as np
import os, sys
from PARAMETERS import *


TRAIN_BATCH_SIZE = 16


with tf.device('/cpu:0'):

	with tf.name_scope('Parameters'):

		with tf.name_scope('Weights'):

			W = {

				1: tf.truncated_normal([3, 3, 3, 1, 32], stddev=0.5),
				2: tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.5),
				3: tf.truncated_normal([3, 3, 3, 32, 64], stddev=0.5),
				4: tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.5),
				5: tf.truncated_normal([3, 1, 1, 64, 128], stddev=0.5),
				6: tf.truncated_normal([3, 1, 1, 128, 128], stddev=0.5),
				7: tf.truncated_normal([3, 1, 1, 128, 256], stddev=0.5),
				8: tf.truncated_normal([3, 1, 1, 256, 256], stddev=0.5),
				9: tf.truncated_normal([32768, 1024], stddev=0.5),
				10: tf.truncated_normal([1024, 64], stddev=0.5),
				11: tf.truncated_normal([64, 1], stddev=0.5)

			}

		with tf.name_scope('Biases'):
			B = {

				1: tf.random_normal([32]),
				2: tf.random_normal([32]),
				3: tf.random_normal([64]),
				4: tf.random_normal([64]),
				# 5: tf.random_normal([128]),
				# 6: tf.random_normal([128]),
				# 7: tf.random_normal([256]),
				# 8: tf.random_normal([256]),
				9: tf.random_normal([1024]),
				10: tf.random_normal([64]),
				11: tf.random_normal([1])

			}

		beta1 = tf.Variable(0.0, [100, 7, 7, 32])
		gamma1 = tf.Variable(1.0, [100, 7, 7, 32])
		beta2 = tf.Variable(0.0, [50, 3, 3, 64])
		gamma2 = tf.Variable(1.0, [50, 3, 3, 64])
		# beta3 = tf.Variable(0.0, [25, 3, 3, 128])
		# gamma3 = tf.Variable(1.0, [25, 3, 3, 128])
		# beta4 = tf.Variable(0.0, [12, 3, 3, 256])
		# gamma4 = tf.Variable(1.0, [12, 3, 3, 256])



def predict(x, W, B):

	with tf.name_scope('Model'):

		with tf.name_scope('Layer1'):
			conv1 = tf.nn.conv3d(x, W[1], strides=[1, 1, 1, 1, 1], padding='SAME') + B[1]
			# conv1 = tf.nn.conv3d(conv1, W[2], strides=[1, 1, 1, 1, 1], padding='SAME') + B[2]
			relu1 = tf.nn.relu(conv1)

			pool1 = tf.nn.max_pool3d(relu1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
			mean1, variance1 = tf.nn.moments(pool1, axes=[0, 2, 3], keep_dims=True)
			norm1 = tf.nn.batch_normalization(pool1, mean=mean1, variance=variance1, offset=beta1, scale=gamma1, variance_epsilon=1e-5)


		with tf.name_scope('Layer2'):
			conv2 = tf.nn.conv3d(norm1, W[3], strides=[1, 1, 1, 1, 1], padding='SAME') + B[3]
			# conv2 = tf.nn.conv3d(conv2, W[4], strides=[1, 1, 1, 1, 1], padding='SAME') + B[4]
			relu2 = tf.nn.relu(conv2)

			pool2 = tf.nn.max_pool3d(relu2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
			mean2, variance2 = tf.nn.moments(pool2, axes=[0, 2, 3], keep_dims=True)
			norm2 = tf.nn.batch_normalization(pool2, mean=mean2, variance=variance2, offset=beta2, scale=gamma2, variance_epsilon=1e-5)


		# with tf.name_scope('Layer3'):
		# 	conv3 = tf.nn.conv3d(norm2, W[5], strides=[1, 1, 1, 1, 1], padding='SAME') + B[5]
		# 	conv3 = tf.nn.conv3d(conv3, W[6], strides=[1, 1, 1, 1, 1], padding='SAME') + B[6]
		# 	relu3 = tf.nn.relu(conv3)

		# 	pool3 = tf.nn.max_pool3d(relu3, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
		# 	mean3, variance3 = tf.nn.moments(pool3, axes=[0, 2, 3], keep_dims=True)
		# 	norm3 = tf.nn.batch_normalization(pool3, mean=mean3, variance=variance3, offset=beta3, scale=gamma3, variance_epsilon=1e-5)


		# with tf.name_scope('Layer4'):
		# 	conv4 = tf.nn.conv3d(norm3, W[7], strides=[1, 1, 1, 1, 1], padding='SAME') + B[7]
		# 	conv4 = tf.nn.conv3d(conv4, W[8], strides=[1, 1, 1, 1, 1], padding='SAME') + B[8]
		# 	relu4 = tf.nn.relu(conv4)

		# 	pool4 = tf.nn.max_pool3d(relu4, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
		# 	mean4, variance4 = tf.nn.moments(pool4, axes=[0, 2, 3], keep_dims=True)
		# 	norm4 = tf.nn.batch_normalization(pool4, mean=mean4, variance=variance4, offset=beta4, scale=gamma4, variance_epsilon=1e-5)


		with tf.name_scope('Flatten'):
			dim = tf.reduce_prod(norm2.get_shape().as_list()[1:])
			flatten = tf.reshape(norm2, [-1, dim])


		with tf.name_scope('Fully_Connected_Layer_1'):
			fc1 = tf.add(tf.matmul(flatten, W[9]), B[9])
			fc1 = tf.nn.sigmoid(fc1)


		with tf.name_scope('Fully_Connected_Layer_2'):
			fc2 = tf.add(tf.matmul(fc1, W[10]), B[10])
			fc2 = tf.nn.sigmoid(fc2)


		with tf.name_scope('Output'):
			out = tf.add(tf.matmul(fc2, W[11]), B[11])
			out = tf.nn.sigmoid(out)


		return tf.clip_by_value(out, 1e-3, 1.0)
# end of predict()


def Generator(path_to_kaggle_dataset):
	all_positive_samples = os.listdir(os.path.join(path_to_kaggle_dataset, 'Cancer'))
	all_negative_samples = os.listdir(os.path.join(path_to_kaggle_dataset, 'Non Cancer'))

	counter_positive = 0
	counter_negative = 0

	new_data = True

	while new_data:
		start_pos_positive = counter_positive
		end_pos_positive = counter_positive + (TRAIN_BATCH_SIZE / 2)
		start_pos_negative = counter_negative
		end_pos_negative = counter_negative + (TRAIN_BATCH_SIZE / 2)

		positive_sample_names = np.array(all_positive_samples[start_pos_positive: end_pos_positive])
		negative_sample_names = np.array(all_negative_samples[start_pos_negative: end_pos_negative])

		if not len(positive_sample_names) == len(negative_sample_names):
			break

		batch_sample_names = np.array(np.concatenate([positive_sample_names, negative_sample_names]))

		counter_positive = end_pos_positive
		counter_negative = end_pos_negative

		batch = []

		for i, each_sample_name in enumerate(batch_sample_names):
			folder = 'Cancer' if i<(TRAIN_BATCH_SIZE/2) else 'Non Cancer'
			path_sample = os.path.join(path_to_kaggle_dataset, folder, each_sample_name)
			sample_data = np.load(path_sample).item()
			sample_data_img = sample_data['data']
			# print sample_data_img.shape
			label = 1 if folder == 'Cancer' else 0
			batch.append([ sample_data_img, label ])

		np.random.shuffle(batch)
		batch = np.array(batch)

		if not batch.shape == (TRAIN_BATCH_SIZE, 2):
			new_data = False
			break
		else:
			yield batch
# end Generator()


def train(path_to_kaggle_dataset, path_to_folder_for_storing_checkpoint):
	saver = tf.train.Saver()

	with tf.name_scope('Placeholders'):
		x = tf.placeholder(tf.float32, [None, 125, 15, 15, 1], name='x')
		y = tf.placeholder(tf.float32, name='y')

	prediction = predict(x, W, B)

	with tf.name_scope('Loss'):
		regularization = tf.nn.l2_loss(W[1]) + tf.nn.l2_loss(W[3]) \
					   + tf.nn.l2_loss(W[9]) \
					   + tf.nn.l2_loss(W[10]) + tf.nn.l2_loss(W[11])
		cost = tf.reduce_sum(tf.losses.log_loss(labels=y, predictions=prediction)) + (1e-5 * regularization)
		tf.summary.scalar('Loss', cost)

	with tf.name_scope('Oprtimizer'):
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			optimizer = tf.train.AdamOptimizer().minimize(cost)

	saver = tf.train.Saver(tf.global_variables())

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)

		merged = tf.summary.merge_all()
		log_summary = tf.summary.FileWriter('.', sess.graph)

		counter = 0

		for epoch in range(50):
			generator = Generator(path_to_kaggle_dataset)

			loss_history = []

			for count, batch in enumerate(generator):
				counter += 1

				
				batch_x = np.array([X for X, _ in batch]).reshape([TRAIN_BATCH_SIZE, 125, 15, 15, 1])
				batch_x = batch_x.astype(np.float32)
				batch_y = np.array([Y for _, Y in batch]).reshape([TRAIN_BATCH_SIZE, 1])
				batch_y = batch_y.astype(np.float32)

				_summary, p, l, _ = sess.run([merged, prediction, cost, optimizer], feed_dict={x: batch_x, y: batch_y})
				log_summary.add_summary(_summary, counter)
				loss_history.append(float(l))

				print p
				print batch_y

				print 'Iteration: {0}\tLoss: {1}'.format(counter, l)

				if not len(batch) == TRAIN_BATCH_SIZE:
					break
				
			os.system('mkdir {}/Epoch_{}'.format(path_to_folder_for_storing_checkpoint, epoch+1))
			save_path = saver.save(sess, '{}/Epoch_{}/Model_checkpoint.ckpt'.format(path_to_folder_for_storing_checkpoint, epoch+1))
			print 'Saved model to {}'.format(save_path)
# end train()


if __name__ == '__main__':
	args = sys.argv
	train(args[1], args[2])












