'''
Author: Aman Rana
Contact: arana@wpi.edu
Website: http://www,amanrna.com

Topic: This code tests preprocessing the lung cancer image by trying certain augmentations.
'''

# imports
import numpy as np
import SimpleITK as itk
import os, cv2
import matplotlib.pyplot as plt


file_name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.mhd'

img_data = itk.ReadImage(file_name)
img = itk.GetArrayFromImage(img_data)

img1 = img[100, :, :]

img1 = img1 < -700

plt.imshow(img1, cmap='gray')
plt.show()