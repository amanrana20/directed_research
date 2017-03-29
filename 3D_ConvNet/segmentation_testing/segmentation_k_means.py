import os, cv2
import numpy as np
import numpy as np
import SimpleITK as itk
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


PATH_BASE = '/home/amanrana/kaggle_main/Data Science Bowl Kaggle/dataset/Annotated Lung Cancer Dataset/Train_Data3/Cancer'
files = os.listdir(PATH_BASE)

num_training_samples = 5
# num_testing_samples = 100
# n_th_slice_to_examine = np.random.randint(0, num_testing_samples)

x = np.zeros([num_training_samples, 128, 128])
# y = np.zeros([num_testing_samples, 128, 128])

for i, f in enumerate(files[: num_training_samples]):
	data = np.load(os.path.join(PATH_BASE, f))
	x[i, :, :] = data[1, :, :].reshape([128, 128])

# for j, f in enumerate(files[num_training_samples: num_training_samples+num_testing_samples]):
# 	data = np.load(os.path.join(PATH_BASE, f))
# 	y[j, :, :] = data[1, :, :].reshape([128, 128])

fig = plt.figure()

for i in range(num_training_samples):
	fig.add_subplot(2,  num_training_samples, (2*i+1))
	plt.imshow(x[i, :, :].reshape([128, 128]), cmap='gray')

	x1 = x[i, :, :].reshape([-1, 1])

	pred = KMeans(n_clusters = 2).fit_predict(x1)
	pred = pred.reshape([128, 128])

	fig.add_subplot(2, num_training_samples, i*2+2)
	plt.imshow(pred, cmap='gray')


plt.show()
