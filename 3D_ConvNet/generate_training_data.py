'''
Author: Aman Rana
Contact: arana@wpi.edu

Topic: This file contains code to generate small 3D volumes around the annotated cancer nodules (positives) and some random 3D volumnes (negatives), while trying to maintain positive to negavive sample ratio.
'''

# Import starements
import numpy as np # for creating 3D numpy arrays
import pandas as pd # reading cs files ad processing
import os, cv2, random
import SimpleITK as itk
import matplotlib.pyplot as plt
from PARAMETERS import *

# Some variables
PATH_BASE = '../../../kaggle_main/Data Science Bowl Kaggle/dataset/Annotated Lung Cancer Dataset'
PATH_DATASET = os.path.join(PATH_BASE, 'annotated_data')
CSV_CANDIDATES = os.path.join(PATH_BASE, 'csv files/candidates.csv')


## Some Variables
N_SLICES_PER_SAMPLE = NUM_SLICES


# Generated Training data path
PATH_GENERATED_TRAINING_DATA = os.path.join(PATH_BASE, PATH_TO_GENERATE_TRAINING_DATA)


# reading data from canditates.csv, which contains possible nodule candidates - both positive and negative
candidates_data = pd.read_csv(CSV_CANDIDATES)


## Printing some stats
print 'Number of data pairs in the Dataset: {}\n'.format(len(os.listdir(PATH_DATASET))/2)
num_samples = candidates_data['seriesuid']
print 'Number of data rows in the annotated.csv file: {}\n'.format(len(num_samples))
num_cancer_cases = len(candidates_data[candidates_data['class'] == 1])
num_non_cancer_cases = len(candidates_data[candidates_data['class'] == 0])
print 'Ratio of non-canecr to cancer samples: {} / {} = {}\n'.format(num_non_cancer_cases, num_cancer_cases, float(num_non_cancer_cases) / float(num_cancer_cases))


def convert_to_voxel(x, y, z, origin, spacing):
	return abs(np.rint((np.array([x, y, z]) - np.array(origin)) / spacing).astype('int'))


def get_preprocessed_image(im):
	for i in range(NUM_SLICES):
		sl = im[i].copy()
		# sl = cv2.GaussianBlur(sl, (15, 15), 0)
		sl[sl > -300] = 255
		sl[sl < -300] = 0
		sl1 = np.uint8(sl)

		# find surrounding torso from the threshold and make a mask
		contours, _ = cv2.findContours(sl1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		largest_contour = max(contours, key=cv2.contourArea)
		mask = np.zeros(sl.shape, np.uint8)
		cv2.fillPoly(mask, [largest_contour], 255)

		# apply mask to threshold image to remove outside. this is our new mask
		sl = ~sl
		sl[(mask == 0)] = 0

		# apply closing to the mask
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
		sl = cv2.morphologyEx(sl, cv2.MORPH_OPEN, kernel)  # <- to remove speckles...
		sl = cv2.morphologyEx(sl, cv2.MORPH_DILATE, kernel)
		sl = cv2.morphologyEx(sl, cv2.MORPH_DILATE, kernel)
		sl = cv2.morphologyEx(sl, cv2.MORPH_CLOSE, kernel)
		sl = cv2.morphologyEx(sl, cv2.MORPH_CLOSE, kernel)
		sl = cv2.morphologyEx(sl, cv2.MORPH_ERODE, kernel)
		sl = cv2.morphologyEx(sl, cv2.MORPH_ERODE, kernel)

		f_im = im[i].copy()
		f_im[sl == 0] = 0
		f_im[sl < -1] = 0

		im[i] = f_im

	return im


def save_sample_to_disk(sample):
	selected_folder = 'Cancer' if sample['class'] else 'Non Cancer'
	sample_save_path = os.path.join(PATH_GENERATED_TRAINING_DATA, selected_folder, '{}'.format(sample['name']))
	np.save(sample_save_path, sample['data'])


def get_valid_frame(x, y, r1, r2):
	a = IMAGE_SIZE-r1
	b = IMAGE_SIZE-r2

	fr1 = ((x-r1, y-r2), (x+a, y-r2), (x+a, y+b), (x-r1, y+b))
	fr2 = ((x-a, y-r2), (x+r1, y-r2), (x+r1, y+b), (x-a, y+b))
	fr3 = ((x-r1, y-b), (x+a, y-b), (x+a, y+r2), (x-r1, y+r2))
	fr4 = ((x-a, y-b), (x+r1, y-b), (x+r1, y+r2), (x-a, y+r2))
	
	all_possible_frames = [fr1, fr2, fr3, fr4]
	random.shuffle(all_possible_frames)

	for i, each_frame in enumerate(all_possible_frames):
		frame_validity = []

		for ptX, ptY in each_frame:
			if ptX <= 0 or ptX >= SCAN_CROSS_SECTION_SIZE or ptY <= 0 or ptY >= SCAN_CROSS_SECTION_SIZE:
				frame_validity.append(False)
			else:
				frame_validity.append(True)

		if False not in frame_validity:
			return all_possible_frames[i]

	return False


def make_valid_sample(img, x, y, cls):
	subset_valid = False
	lim = int(0.9 * IMAGE_SIZE)

	valid_frame = None

	while not subset_valid:
		r1 = np.random.randint(5, lim)
		r2 = np.random.randint(5, lim)

		valid_frame = get_valid_frame(x, y, r1, r2)

		if valid_frame:
			subset_valid = True
		else:
			if lim < IMAGE_SIZE - 5:
				lim	-= 1
			subset_valid = False

	p1, p2, p3, p4 = valid_frame

	return img[:, p1[1]: p3[1], p1[0]: p3[0]]


def process_samples(img, x, y, z, i, cancer_cls):
	num_samples_wanted = NUM_SAMPLES_PER_POSITIVE_SAMPLE_PER_SAMPLE if cancer_cls else 1
	name = '{}_{}_{}_{}'.format(i, z, y, x) if cancer_cls else '{}'.format(i)

	for j in range(num_samples_wanted):
		sample = {
					'data': make_valid_sample(img, x, y, cancer_cls),
					'name': '{}_{}'.format(name, j) if cancer_cls else name,
					'class': cancer_cls
				 }

		save_sample_to_disk(sample)


def is_cancer(cls):
	return True if cls == 1 else 0


# Creating the datase
def create_dataset():
	for i in range(len(candidates_data.index)):
		print 'Row:', i+1

		row = candidates_data.iloc[i]
		scan_id, posX, posY, posZ, cancer_class = row
		cancer_class = int(cancer_class)

		# Scan data
		scan = itk.ReadImage(os.path.join(PATH_DATASET, scan_id + '.mhd'))
		img = itk.GetArrayFromImage(scan)

		# Getting the origin and spacing for conversion to voxel
		origin = scan.GetOrigin()
		spacing = scan.GetSpacing()

		# Converting the candidate nodule coordinates to voxel
		posX, posY, posZ = convert_to_voxel(posX, posY, posZ, origin, spacing)

		isCancer = is_cancer(cancer_class)

		original = img[posZ-1: posZ+2, :, :]

		# Segmenting the lug tissue
		segmented_img = get_preprocessed_image(original)
		
		try:
			process_samples(segmented_img, posX, posY, posZ, i, isCancer)
		except:
			print 'Problem in', i

		
create_dataset()
