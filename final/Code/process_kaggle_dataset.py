'''
Author: Aman Rana
Contact: arana@wpi.edu

Title: This file contains code for inference using the trained model for the Lung Cancer Detection Challenge hosted on Kaggle.com
'''

import os, cv2, sys
import dicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import tensorflow as tf
from PARAMETERS import *


stride = [32, 32, 5]


def get_3D_SCAN(scan_name, slices_dcm, path_to_kaggle_dataset):
	each_scan_data = []
	for i, each_dcm in enumerate(slices_dcm):
		slice_data = dicom.read_file((os.path.join(path_to_kaggle_dataset, scan_name, slices_dcm[i])))
		each_scan_data.append(slice_data)
	try:
		each_scan_data.sort(key = lambda z: int(z.SliceLocation))
	except:
		each_scan_data.sort(key = lambda z: int(z.InstanceNumber))

	scan = np.zeros([len(slices_dcm), SCAN_CROSS_SECTION_SIZE, SCAN_CROSS_SECTION_SIZE]).astype(np.float32)

	for i in range(len(each_scan_data)):
		scan[i, :, :] = (each_scan_data[i].pixel_array * each_scan_data[i].RescaleSlope) + each_scan_data[i].RescaleIntercept

	return scan
# end get_3D_SCAN()


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
# end CPU variable inititlization


def predict(x, W, B, beta1, gamma1, beta2, gamma2, beta3, gamma3):

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
# end predict()



def generator(csv, scan_names, path_to_kaggle_dataset):
	for i, scan_name in enumerate(scan_names):
		slices_dcm = os.listdir(os.path.join(path_to_kaggle_dataset, scan_name))
		scan = np.array(get_3D_SCAN(scan_name, slices_dcm, path_to_kaggle_dataset)).astype(np.float32)
		labl = int(csv.loc[csv.id == scan_name]['cancer'])
		print 'Cancer' if labl == 1 else 'Non Cancer'
		limZ, limY, limX = scan.shape

		pos_Z = 0
		for z in range(0, limZ, stride[2]):
			sub_VOLS = []
			for y in range(0, limY, stride[1]):
				for x in range(0, limX, stride[0]):
					x_st, x_end = x, x + IMAGE_SIZE
					y_st, y_end = y, y + IMAGE_SIZE
					z_st, z_end = z, z + NUM_SLICES

					sub_vol = scan[z_st: z_end, y_st: y_end, x_st: x_end]
					sub_vol_cp = np.array(sub_vol).copy()

					if sub_vol_cp.shape != (10, 64, 64):
						continue

					sub_VOLS.append(sub_vol)
			if len(sub_VOLS) == 225:
				yield i, scan_name, labl, pos_Z, sub_VOLS
			else:
			 continue
			
			pos_Z += 1
# end generator()



def create_mid_data(path_to_kaggle_dataset, path_to_labels_csv, path_to_store_proecssed_kaggle_dataset, path_to_trained_model):
	csv = pd.read_csv(path_to_labels_csv)
	scan_names = list(csv.id)
	GEN = generator(csv, scan_names, path_to_kaggle_dataset)

	with tf.name_scope('Placeholders'):
		x = tf.placeholder(tf.float32, shape=[None, NUM_SLICES, IMAGE_SIZE, IMAGE_SIZE, 1], name='X')
		y = tf.placeholder(tf.float32, name='Y')

	model_output = predict(x, W, B, beta1, gamma1, beta2, gamma2, beta3, gamma3)
	saver = tf.train.Saver()
	
	with tf.Session() as sess:
		saver.restore(sess, path_to_trained_model)
		lastI = 0
		last_scan_name = ''
		prediction_log = {}
		for i, scan_name, label, posZ, vols in GEN:
			if lastI != i:
				print i
				folder = 'Cancer' if label == 1 else 'Non Cancer'
				np.save('{}/{}/{}'.format(path_to_store_proecssed_kaggle_dataset, folder, scan_name), prediction_log[last_scan_name])
				prediction_log = {}
				lastI = i
			else:
				if scan_name not in prediction_log.keys():
					prediction_log[scan_name] = {'data': np.zeros([125, 15, 15]).astype(np.float32), 'label': label}

				model_prediction = sess.run(model_output, feed_dict={x: np.array(vols).reshape([225, NUM_SLICES, IMAGE_SIZE, IMAGE_SIZE, 1])})
				cancer_prob = np.array(model_prediction[:, 0]).reshape([15, 15])
				print 'Slice: {}'.format(posZ)

				prediction_log[scan_name]['data'][posZ, :, :] = cancer_prob

				last_scan_name = scan_name

			
# end of create_mid_data

if __name__ == '__main__':
	args = sys.argv
	create_mid_data(args[1], args[2], args[3], args[4])


