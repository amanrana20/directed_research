import tensorflow as tf
import numpy as np
import pandas as pd
import os, sys


INFERENCE_BATCH_SIZE = 32
NUM_SLICES = 200
IMAGE_SIZE = 64



with tf.device('/cpu:0'):

	with tf,name_scope('Parameters'):

		with tf.name_scope('Weights'):

			W = {

				1: tf.truncated_normal([3, 3, 3, 3, 32], stddev=0.5),
				2: tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.5),
				3: tf.truncated_normal([3, 3, 3, 32, 64], stddev=0.5),
				4: tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.5),
				5: tf.truncated_normal([3, 1, 1, 64, 128], stddev=0.5),
				6: tf.truncated_normal([3, 1, 1, 128, 128], stddev=0.5),
				7: tf.truncated_normal([3, 1, 1, 128, 256], stddev=0.5),
				8: tf.truncated_normal([3, 1, 1, 256, 256], stddev=0.5),
				9: tf.truncated_normal([27648, 1024], stddev=0.5),
				10: tf.truncated_normal([1024, 64], stddev=0.5),
				11: tf.truncated_normal([64, 2], stddev=0.5)

			}

		with tf.name_scope('Biases'):
			B = {

				1: tf.random_normal([32]),
				2: tf.random_normal([32]),
				3: tf.random_normal([64]),
				4: tf.random_normal([64]),
				5: tf.random_normal([128]),
				6: tf.random_normal([128]),
				7: tf.random_normal([256]),
				8: tf.random_normal([256]),
				9: tf.random_normal([1024]),
				10: tf.random_normal([64]),
				11: tf.random_normal([2])

			}

		beta1 = tf.Variable(0.0, [100, 7, 7, 32])
		gamma1 = tf.Variable(1.0, [100, 7, 7, 32])
		beta2 = tf.Variable(0.0, [50, 3, 3, 64])
		gamma2 = tf.Variable(1.0, [50, 3, 3, 64])
		beta3 = tf.Variable(0.0, [25, 3, 3, 128])
		gamma3 = tf.Variable(1.0, [25, 3, 3, 128])
		beta4 = tf.Variable(0.0, [12, 3, 3, 256])
		gamma4 = tf.Variable(1.0, [12, 3, 3, 256])



def predict(x):

	with tf.name_scope('Model'):

		with tf.name_scope('Layer1'):
			conv1 = tf.nn.conv3d(x, W[1], stride=[1, 1, 1, 1, 1], padding='SAME') + B[1]
			conv1 = tf.nn.conv3d(conv1, W[2], stride=[1, 1, 1, 1, 1], padding='SAME') + B[2]
			relu1 = tf.nn.relu(conv1)

			pool1 = tf.nn.max_pool3d(relu1, k_size=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
			mean1, variance1 = tf.nn.moments(pool1, axes=[0, 2, 3], keep_dims=True)
			norm1 = tf.nn.batch_normalization(pool1, mean=mean1, variance=variance1, offset=beta1, scale=gamma1, variance_epsilon=1e-5)


		with tf.name_scope('Layer2'):
			conv2 = tf.nn.conv3d(norm1, W[3], stride=[1, 1, 1, 1, 1], padding='SAME') + B[3]
			conv2 = tf.nn.conv3d(conv2, W[4], stride=[1, 1, 1, 1, 1], padding='SAME') + B[4]
			relu2 = tf.nn.relu(conv2)

			pool2 = tf.nn.max_pool3d(relu2, k_size=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
			mean2, variance2 = tf.nn.moments(pool2, axes=[0, 2, 3], keep_dims=True)
			norm2 = tf.nn.batch_normalization(pool2, mean=mean2, variance=variance2, offset=beta2, scale=gamma2, variance_epsilon=1e-5)


		with tf.name_scope('Layer3'):
			conv3 = tf.nn.conv3d(norm2, W[5], stride=[1, 1, 1, 1, 1], padding='SAME') + B[5]
			conv3 = tf.nn.conv3d(conv3, W[6], stride=[1, 1, 1, 1, 1], padding='SAME') + B[6]
			relu3 = tf.nn.relu(conv2)

			pool3 = tf.nn.max_pool3d(relu3, k_size=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
			mean3, variance3 = tf.nn.moments(pool3, axes=[0, 2, 3], keep_dims=True)
			norm3 = tf.nn.batch_normalization(pool3, mean=mean3, variance=variance3, offset=beta3, scale=gamma3, variance_epsilon=1e-5)


		with tf.name_scope('Layer4'):
			conv4 = tf.nn.conv3d(norm3, W[7], stride=[1, 1, 1, 1, 1], padding='SAME') + B[7]
			conv4 = tf.nn.conv3d(conv4, W[8], stride=[1, 1, 1, 1, 1], padding='SAME') + B[8]
			relu4 = tf.nn.relu(conv4)

			pool4 = tf.nn.max_pool3d(relu4, k_size=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
			mean4, variance4 = tf.nn.moments(pool4, axes=[0, 2, 3], keep_dims=True)
			norm4 = tf.nn.batch_normalization(pool4, mean=mean4, variance=variance4, offset=beta4, scale=gamma4, variance_epsilon=1e-5)


		with tf.name_scope('Flatten'):
			dim = tf.reduce_prod(norm4.get_shape().as_list()[1:])
			flatten = tf.reshape(norm4, [-1, dim])


		with tf.name_scope('Fully Connected Layer 1'):
			fc1 = tf.add(tf.matmul(flatten, W[9]), B[9])
			fc1 = tf.nn.relu(fc1)


		with tf.name_scope('Fully Connected Layer 2'):
			fc2 = tf.add(tf.matmul(fc1, W[10]), B[10])
			fc2 = tf.nn.relu(fc2)


		with tf.name_scope('Output'):
			out = tf.add(tf.matmul(fc2, W[11]), B[11])
			out = tf.nn.relu(out)


		return tf.clip_by_value(out, 1e-3, 1.0)
# end of predict()



def main(url_to_trained_model_on_kaggle_dataset, url_test_dataset, url_to_store_submission_csv):

	files = os.listdir(url_test_dataset)

	with tf.name_scope('Placeholders'):
		x = tf.placeholder(tf.float32, [None, NUM_SLICES, IMAGE_SIZE, IMAGE_SIZE, 1], name='x')
		y = tf.placeholder(tf.float32, name='y')

	prediction = predict(x)#, W, B, beta1, beta2, beta3, beta4, gamma1, gamma2, gamma3, gamma4
	saver = tf.train.Saver()

	log_predictions = {}

	with tf.Session() as sess:
		saver.restore(sess, url_to_trained_model_on_kaggle_dataset)

		for i, file_name in enumerate(files):
			data = np.load(os.path.join(url_test_dataset, file_name))
			data = data.reshape([1, NUM_SLICES, IMAGE_SIZE, IMAGE_SIZE, 1])
			data = data.astype(np.float32)
			model_prediction = sess.run(prediction, feed_dict={x: data})

			log_predictions[file_name] = model_prediction[0][0]

		submission = pd.DataFrame(log_predictions.items(), columns=['id', 'cancer'])

	submission.to_csv(url_to_store_submission_csv)




if __name__ == '__main__':
	args = sys.argv
	main(args[1], args[2], argv[3])
