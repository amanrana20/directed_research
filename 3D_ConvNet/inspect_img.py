import SimpleITK as itk
import numpy as np
import cv2, os
import matplotlib.pyplot as plt

path = '../../../kaggle_main/Data Science Bowl Kaggle/dataset/Annotated Lung Cancer Dataset/annotated_data'
file_name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.107351566259572521472765997306.mhd'

data = itk.ReadImage(os.path.join(path, file_name))
img = itk.GetArrayFromImage(data)

# plt.imshow(img[89, 150:350, 250:450])  # [zxy]
plt.imshow(img[96, :, :]) 
plt.show()