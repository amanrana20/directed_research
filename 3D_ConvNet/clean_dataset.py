import os
import numpy as np


PATH_BASE = '../../../kaggle_main/Data Science Bowl Kaggle/dataset/Annotated Lung Cancer Dataset/Train_Data'
EXTENSIONS = ['Non Cancer']


for extension in EXTENSIONS:

	for _file in list(os.listdir(os.path.join(PATH_BASE, extension))):

		sample = np.load(os.path.join(PATH_BASE, extension, _file))
		shape = sample.shape

		if shape != (3, 128, 128):

			print shape
			path = os.path.join('~/kaggle_main/Data\ Science\ Bowl\ Kaggle/dataset/Annotated\ Lung\ Cancer\ Dataset/Train_Data', 'Non\ Cancer', _file)
			print path
			os.system('rm {}'.format(path))
			print '\n'