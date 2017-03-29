'''
Author: Aman Rana
Contact: arana@wpi.edu
'''


import numpy as np
import cv2, os
from PARAMETERS import *


## File Paths
PATH_BASE = '../../../kaggle_main/Data Science Bowl Kaggle/dataset/Annotated Lung Cancer Dataset/Train_Data'

## Some parameters
new_data = True  # This variable tells whethe there is more data or not
num_positive_samples = BATCH_SIZE / 2
num_negative_samples = BATCH_SIZE / 2

## Create batch generator
def batch_generator():

	'''
	Since the ratio of positive to negative samples is very low, 
	there is a huge class imbalance. This batch generator creates 
	a batch while preserving the class ratio in the batch.

	Return: (batch)
	'''

	all_positive_samples = os.listdir(os.path.join(PATH_BASE, 'Cancer'))
	all_negative_samples = os.listdir(os.path.join(PATH_BASE, 'Non Cancer'))

	counter_positive = 0
	counter_negative = 0
	
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
			
			path_sample = os.path.join(PATH_BASE, folder, each_sample_name)
			sample_data = np.load(path_sample)

			label = np.array([1, 0]) if folder == 'Cancer' else np.array([0, 1])

			batch.append([sample_data, label])

		np.random.shuffle(batch)

		yield batch