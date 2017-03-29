import os
import numpy as np
from PARAMETERS import *


PATH_BASE = "/home/amanrana/kaggle_main/Data Science Bowl Kaggle/dataset/Annotated Lung Cancer Dataset/Train_Data3"
EXTENSIONS = os.listdir(PATH_BASE)


for extension in EXTENSIONS:

	for _file in list(os.listdir(os.path.join(PATH_BASE, extension))):

		sample = np.load(os.path.join(PATH_BASE, extension, _file))
		shape = sample.shape

		print _file

		if shape != (NUM_SLICES, IMAGE_SIZE, IMAGE_SIZE):

			print shape
			print extension
			# if extension == 'Non Cancer':
			# 	extension = 'Non\ Cancer'
			path = os.path.join(PATH_BASE, extension, _file)
			print path
			os.remove(path)
			print '\n'