'''
Author: Aman Rana
Contact: arana@wpi.edu

Topic: This file contains code to generate small 3D volumes around the annotated cancer nodules (positives) and some random 3D volumnes (negatives), while trying to maintain positive to negavive sample ratio.
'''

# Import starements
import numpy as np # for creating 3D numpy arrays
import pandas as pd # reading cs files ad processing
import os
import SimpleITK as itk
import matplotlib.pyplot as plt

# Some variables
PATH_BASE = '../../../kaggle_main/Data Science Bowl Kaggle/dataset/Annotated Lung Cancer Dataset'
PATH_DATASET = os.path.join(PATH_BASE, 'annotated_data')
CSV_CANDIDATES = os.path.join(PATH_BASE, 'csv files/candidates.csv')


# Generated Training data path
PATH_GENERATED_TRAINING_DATA = os.path.join(PATH_BASE, 'Train_Data')

# reading data from canditates.csv
candidates_data = pd.read_csv(CSV_CANDIDATES)

## Priting some stats
print 'Number of data pairs in the Dataset: {}\n'.format(len(os.listdir(PATH_DATASET))/2)
num_samples = candidates_data['seriesuid']
print 'Number of data rows in the annotated.csv file: {}\n'.format(len(num_samples))

num_cancer_cases = len(candidates_data[candidates_data['class'] == 1])
num_non_cancer_cases = len(candidates_data[candidates_data['class'] == 0])
print 'Ratio of non-canecr to cancer samples: {} / {} = {}\n'.format(num_non_cancer_cases, num_cancer_cases, 1.0 * num_non_cancer_cases / num_cancer_cases)


def convert_to_voxel(x, y, z, origin, spacing):
	return np.rint((np.array([x, y, z]) - origin) / spacing).astype('int')


# Creating the datase
def create_dataset():

	for i in range(len(candidates_data.index)):

		row = candidates_data.iloc[i]
		scan_id, posX, posY, posZ, cancer_class = row

		# Scan data
		scan = itk.ReadImage(os.path.join(PATH_DATASET, scan_id + '.mhd'))
		img = itk.GetArrayFromImage(scan)

		# Getting the origin and spacing for conversion to voxel
		origin = scan.GetOrigin()
		spacing = scan.GetSpacing()
		posX, posY, posZ = convert_to_voxel(posX, posY, posZ, origin, spacing)

		new_3D_sample = img[posZ - 1: posZ + 1, :, :]
		selected_folder = 'Cancer' if cancer_class == 1 else 'Non Cancer'
		sample_save_path = os.path.join(PATH_GENERATED_TRAINING_DATA, selected_folder, '{}'.format(i))
		np.save(sample_save_path, new_3D_sample)

		
		
create_dataset()