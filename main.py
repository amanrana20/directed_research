import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, cv2
import dicom
import scipy.ndimage
import CNN


# Setting the global parameters
PATH_BASE = 'data/'
print('\n{} {}'.format('Files in data folder:', os.listdir(PATH_BASE)))
EXT_DATASET = 'stage1'
EXT_STAGE1_LABELS = 'stage1_labels.csv'
EXT_SAMPLE_SUBMISSION = 'stage1_sample_submission.csv'
patients_names = os.listdir(os.path.join(PATH_BASE, EXT_DATASET))
print('Number of Patients: {}'.format(len(patients_names)))


# VARIABLES
generator_batch_size = 1
TRAINING_BATCH_SIZE = 1
CONSTANT_3D_IMAGE_SHAPE = (512, 512, 512)
NUMBER_OF_TRAINING_ITERATIONS = 1


# Reading the ground truth labels and getting the ratio of both classes
csv_ground_truth = pd.read_csv(os.path.join(PATH_BASE, EXT_STAGE1_LABELS))
cancer = len(csv_ground_truth[csv_ground_truth['cancer'] == 1])
no_cancer = len(csv_ground_truth[csv_ground_truth['cancer'] == 0])
print '# cancer cases: {0}\n# non cancre cases: {1}\n'.format(cancer, no_cancer)


print 'NOTE: The number of cancer patients is {0} times more than non-cancer patients. We will have to modify the loss function to accomodate the imblance in training data by giving more importance to cancer cases by a factor of {0}.\n'.format(no_cancer*1./cancer)


# This function generates the batches. Each batch contains <batch_size> patients with their respective scans. All scans are ordered.
def generate_patient_batches(batch_size):
	for count in range(0, generator_batch_size, batch_size):
		patients = {}
		for i, patients_name in enumerate(patients_names[count: count+batch_size]):
		    label = 1 if csv_ground_truth.loc[csv_ground_truth['id'] == patients_name]['cancer'].any() else 0
		    patients[patients_name] = {'label': label}
		yield patients


def create_patient_data(patient, label):
	path_patient= os.path.join(PATH_BASE, EXT_DATASET, patient)
	patient_slices = os.listdir(path_patient)

	dcms = []
	for slice in patient_slices:
		dcm = dicom.read_file(os.path.join(path_patient, slice))
		dcms.append(dcm)
	dcms.sort(key = lambda z: int(z.InstanceNumber))
	image_3d = [np.array(each_dcm.pixel_array).astype(np.float16) for each_dcm in dcms]

	pixel_spacing = dcms[0].PixelSpacing
	slice_thickness = abs(dcms[0].SliceLocation - dcms[1].SliceLocation)
	rescal_intercept = int(dcms[0].RescaleIntercept)
	rescale_slope = int(dcms[0].RescaleSlope)

	return {'3d image': image_3d, 'label': label, 'pixel spacing': pixel_spacing, 'slice thickness': slice_thickness, 'slope': rescale_slope, 'intercept': rescal_intercept}


def process_3d_images(data):
	# Preprocessing
	img = np.array(data['3d image'])
	img[img == -2000.] = 0
	img = (img * data['slope']) + data['intercept']

	# 3D image interpolation
	img = scipy.ndimage.interpolation.zoom(img, [data['slice thickness'], 1., 1.], mode='nearest')

	# Making all 3D images of a constant size
	num_slices = img.shape[0]
	img_const = np.zeros(shape=CONSTANT_3D_IMAGE_SHAPE)
	img_const += img[0, 0, 0]
	for i in range(num_slices):
		pos = i + int((CONSTANT_3D_IMAGE_SHAPE[0] - num_slices) / 2)
		img_const[pos, :, :] = img[i, :, :]

	return np.array(img_const).astype(np.float16)


batch_generator = generate_patient_batches(generator_batch_size)


def training_batch():
	batch = []
	each_batch = batch_generator.next()

	for patient in each_batch.keys():
		patient_data = create_patient_data(patient, each_batch[patient]['label'])
		
		# Making all the images of constant size
		patient_data['3d image'] = np.array(process_3d_images(patient_data)).reshape(1, 512, 512, 512)

		# Making training batch
		batch.append([patient_data['3d image'].astype(np.float16), patient_data['label']])	
	batch = np.array(batch)
	
	return batch


def train_model():
	model = CNN.model((1, 512, 512, 512), 1)
	# print model.summary()
	for training_iteration in range(NUMBER_OF_TRAINING_ITERATIONS):
		batch = training_batch()
		X = np.array([x for x in batch[:, 0]]).reshape(batch.shape[0], 1, 512, 512, 512)
		X = X.astype(np.float16)
		Y = np.array([int(y) for y in batch[:, 1]]).astype(np.float16)

		model.fit(X, Y, batch_size=TRAINING_BATCH_SIZE, nb_epoch=TRAINING_BATCH_SIZE, verbose=1)



train_model()
