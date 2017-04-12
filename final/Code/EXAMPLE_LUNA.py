import SimpleITK as itk
import os, sys, cv2
import matplotlib.pyplot as plt
import numpy as np

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


path = '/home/amanrana/kaggle_main/Data Science Bowl Kaggle/dataset/Annotated Lung Cancer Dataset/annotated_data/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.mhd'

data = itk.ReadImage(path)
scan = itk.GetArrayFromImage(data)
scan = np.array(scan)

processed_scan = segment_lung(scan)

plt.imshow(processed_scan[40, :, :], cmap='bone')
plt.show()