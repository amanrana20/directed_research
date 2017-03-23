'''
Author: Aman Rana
Contact: arana@wpi.edu

Topic: This file contains code to generate small 3D volumes around the annotated cancer nodules (positives) and some random 3D volumnes (negatives), while trying to maintain positive to negavive sample ratio.
'''

# Import starements
import numpy as np # for creating 3D numpy arrays
import pandas as pd # reading cs files ad processing
import os, cv2
import SimpleITK as itk
import matplotlib.pyplot as plt

# Some variables
PATH_BASE = '../../../kaggle_main/Data Science Bowl Kaggle/dataset/Annotated Lung Cancer Dataset'
PATH_DATASET = os.path.join(PATH_BASE, 'annotated_data')
CSV_CANDIDATES = os.path.join(PATH_BASE, 'csv files/candidates.csv')


## Some Variables
IMAGE_SIZE = 128
N_SLICES_PER_SAMPLE = 3
NUM_SAMPLES_PER_POSITIVE_SAMPLE = 400


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
	
	return abs(np.rint((np.array([x, y, z]) - np.array(origin)) / spacing).astype('int'))


def get_preprocessed_image(im):

    # threshold HU > -300
    img[img>-300] = 255
    img[img<-300] = 0
    img = np.uint8(img)
    
    # find surrounding torso from the threshold and make a mask
    im2, contours, _ = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(mask, [largest_contour], 255)
    
    # apply mask to threshold image to remove outside. this is our new mask
    img = ~img
    img[(mask == 0)] = 0 # <-- Larger than threshold value
    
    # apply closing to the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # <- to remove speckles...
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    
    # apply mask to image
    img2 = pat[ int(len(pat)/2) ].image.copy()
    img2[(img == 0)] = -2000 # <-- Larger than threshold value



def save_sample_to_disk(sample):

	selected_folder = 'Cancer' if sample['class'] == 1 else 'Non Cancer'
	sample_save_path = os.path.join(PATH_GENERATED_TRAINING_DATA, selected_folder, '{}'.format(sample['name']))
	np.save(sample_save_path, sample['data'])


def create_multiple_positive_images(img, v_x, v_y, v_z, i, Class):

	for j in range(NUM_SAMPLES_PER_POSITIVE_SAMPLE):

		subset_valid = False
		lim = 100

		while subset_valid == False:

			rx = np.random.randint(0, lim)
			ry = np.random.randint(0, lim)

			subset_valid = True
	
			if v_y-ry < 0 or v_x-rx < 0 or v_y + (IMAGE_SIZE - ry) > 512 or v_x + (IMAGE_SIZE - rx) > 512:

				lim += 1
				subset_valid = False


		# print(v_z, v_y-ry, v_y + (IMAGE_SIZE - ry), v_x-rx, v_x + (IMAGE_SIZE - rx))

		image = img[v_z-1: v_z+2, v_y-ry: v_y + (IMAGE_SIZE - ry), v_x-rx: v_x + (IMAGE_SIZE - rx)]
		# image2 = img[v_z, v_y-ry: v_y + (IMAGE_SIZE - ry), v_x-rx: v_x + (IMAGE_SIZE - rx)]
		# plt.imshow(image2, cmap='gray')
		# plt.show()

		name = '{}_{}_{}_{}_{}'.format(i, v_z, v_y, v_x, j)

		sample = {
					'data': image,
					'name': name,
					'class': Class
				 }

		save_sample_to_disk(sample)


# Creating the datase
def create_dataset():

	for i in range(len(candidates_data.index)):
	# for i in range(3670, 3680):
		print i

		row = candidates_data.iloc[i]
		scan_id, posX, posY, posZ, cancer_class = row
		cancer_class = int(cancer_class)

		# Scan data
		scan = itk.ReadImage(os.path.join(PATH_DATASET, scan_id + '.mhd'))
		img = itk.GetArrayFromImage(scan)


		img = get_preprocessed_image(img)

		# Getting the origin and spacing for conversion to voxel
		origin = scan.GetOrigin()
		spacing = scan.GetSpacing()

		# Converting the candidate nodule coordinates to voxel
		posX, posY, posZ = convert_to_voxel(posX, posY, posZ, origin, spacing)

		if cancer_class == 1:

			create_multiple_positive_images(img, posX, posY, posZ, i, cancer_class)
		
		else:

			sample = {
						'data': img[posZ-1: posZ+2, posY-63: posY+(IMAGE_SIZE-63), posX-63: posX+(IMAGE_SIZE-63)],
						'name': '{}'.format(i),
						'class': cancer_class
			}

			save_sample_to_disk(sample)

		
		
create_dataset()
