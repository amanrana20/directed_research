'''
Author: Aman Rana
Contact: arana@wpi.edu

Topic: This file contains code to generate small 3D volumes around the annotated cancer nodules (positives) and some random 3D volumnes (negatives), while trying to maintain positive to negavive sample ratio.
'''

# Import starements
import numpy as np # for creating 3D numpy arrays
import pandas as pd # reading cs files ad processing
import os, cv2, random, sys
import SimpleITK as itk
import matplotlib.pyplot as plt
from PARAMETERS import *


N_SLICES_PER_SAMPLE = NUM_SLICES


def convert_to_voxel(x, y, z, origin, spacing):
	return abs(np.rint((np.array([x, y, z]) - np.array(origin)) / spacing).astype('int'))


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
	lim = int(0.8 * IMAGE_SIZE)

	valid_frame = None

	while not subset_valid:
		r1 = np.random.randint(int(0.2 * IMAGE_SIZE), lim)
		r2 = np.random.randint(int(0.2 * IMAGE_SIZE), lim)

		valid_frame = get_valid_frame(x, y, r1, r2)

		if valid_frame:
			subset_valid = True
		else:
			if lim < IMAGE_SIZE - 5:
				lim	-= 1
			subset_valid = False

	p1, p2, p3, p4 = valid_frame

	return img[:, p1[1]: p3[1], p1[0]: p3[0]]


def process_samples(img, x, y, z, a, b, has_cancer):
	num_samples_wanted = NUM_SAMPLES_PER_POSITIVE_SAMPLE_PER_SAMPLE if has_cancer else 1
	name = '{}_{}_{}_{}_{}'.format(a, z, y, x, b)

	for j in range(num_samples_wanted):
		try:
			sample = {
						# 'data': get_segmented_img(make_valid_sample(img, x, y, has_cancer)),
						'data': make_valid_sample(img, x, y, has_cancer),
						'name': '{}_{}'.format(name, j),
						'class': has_cancer
					 }
		
			save_sample_to_disk(sample)
		except:
			continue


def is_cancer(cls):
	return True if cls == 1 else False


def is_sample_allowed(z, Z):
	for each_z_loc in Z:
		min = each_z_loc - 25
		max = each_z_loc + 25

		if (min < int(z) < max):
			return False

	return True


def segment_lung(scan):
	s = np.zeros(scan.shape)
	for i in range(scan.shape[0]):
		slice = np.array(scan[i]).reshape(scan.shape[1:])
		slice1 = slice.copy()
		slice1[slice1 < -1100] = 0
		slice1 = cv2.GaussianBlur(slice1, (9, 9), 0)

		slice1 = slice1 < -400
		slice1 = np.uint8(slice1)

		im2, contours, _ = cv2.findContours(slice1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		largest_contour = sorted(contours, key=cv2.contourArea)[-3:]

		mask = np.zeros(slice.shape, np.uint8)
		cv2.fillPoly(mask, largest_contour, 255)

		slice1 = ~slice1
		slice1[mask == 0] = 0

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
		slice1 = cv2.morphologyEx(slice1, cv2.MORPH_OPEN, kernel)  # <- to remove speckles...
		slice1 = cv2.morphologyEx(slice1, cv2.MORPH_DILATE, kernel)
		slice1 = cv2.morphologyEx(slice1, cv2.MORPH_DILATE, kernel)
		slice1 = cv2.morphologyEx(slice1, cv2.MORPH_CLOSE, kernel)
		slice1 = cv2.morphologyEx(slice1, cv2.MORPH_CLOSE, kernel)
		slice1 = cv2.morphologyEx(slice1, cv2.MORPH_ERODE, kernel)
		slice1 = cv2.morphologyEx(slice1, cv2.MORPH_ERODE, kernel)


		slice[slice1 == 0] = 0
		slice[slice > 1000] = 0
		slice[slice <= -1024] = 0
		s[i, :, :] = slice

	return np.array(s)


def prepare_and_process(row, a, b):
	scan_id, posX, posY, posZ, cancer_class = row
	isCancer = is_cancer(int(cancer_class))

	scan = itk.ReadImage(os.path.join(PATH_DATASET, scan_id + '.mhd'))
	img = itk.GetArrayFromImage(scan)
	
	# Getting the origin and spacing for conversion to voxel
	origin = scan.GetOrigin()
	spacing = scan.GetSpacing()

	# Converting the candidate nodule coordinates to voxel
	posX, posY, posZ = convert_to_voxel(posX, posY, posZ, origin, spacing)
	randomZ = np.random.randint(3, 8)
	original = img[posZ-randomZ: posZ+(NUM_SLICES-randomZ), :, :]

	# Segmenting the lug tissue
	try:
		segmented_img = segment_lung(original)
	except:
		print 'Error in segmenting lungs'
		return False
	
	try:
		process_samples(segmented_img, posX, posY, posZ, a, b, isCancer)
	except (RuntimeError, TypeError, NameError) as e:
		print 'Problem in', e
	


# Creating the datase
def create_dataset():
	counter = 1
	scan_names = candidates_data['seriesuid'].unique()
	
	for i in range(0, len(scan_names)):
		scan_name = scan_names[i]
		print 'Scan:', i+1

		df_positive = candidates_data.loc[(candidates_data.seriesuid == scan_name) & (candidates_data['class'] == 1), ]
		df_negative = candidates_data.loc[(candidates_data.seriesuid == scan_name) & (candidates_data['class'] == 0), ]

		df_positive.reset_index()
		df_negative.reset_index()

		positive_z_locations = list(df_positive['coordZ'])

		count_pos = 0
		count_neg = 0


		for j in range(len(df_negative.index)):
			row = df_negative.iloc[j]
			z = row['coordZ']
			
			if is_sample_allowed(z, positive_z_locations):
				count_neg += 1
				resp = prepare_and_process(row, counter, count_neg)
				if resp == False:
					continue

		for k in range(len(df_positive.index)):
			row = df_positive.iloc[k]
			count_pos += 1
			response = prepare_and_process(row, counter, count_pos)
			if response == False:
				continue

		print 'Number of negative samples processed: {}\nNumber of positive samples processed: {}'.format(count_neg, count_pos)


		counter += 1



if __name__ == '__main__':
	args = sys.argv

	global PATH_DATASET 
	PATH_DATASET = args[1]

	global PATH_TO_CSV_CANDIDATES 
	PATH_TO_CSV_CANDIDATES = args[2]

	global PATH_GENERATED_TRAINING_DATA
	PATH_GENERATED_TRAINING_DATA = args[3]

	global candidates_data
	candidates_data = pd.read_csv(PATH_TO_CSV_CANDIDATES)

	create_dataset()