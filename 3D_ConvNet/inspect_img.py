import SimpleITK as itk
import numpy as np
import cv2, os
import matplotlib.pyplot as plt

path = '../../../kaggle_main/Data Science Bowl Kaggle/dataset/Annotated Lung Cancer Dataset/annotated_data'
file_name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.130438550890816550994739120843.mhd'

data = itk.ReadImage(os.path.join(path, file_name))
img = itk.GetArrayFromImage(data)
org = data.GetOrigin()
spacing = data.GetSpacing()
pos = np.array([-143.59, 65.2, -20.28])

v = abs(np.rint(np.array(pos-org)/spacing))
X, Y, Z = v.astype(int)
print v

# plt.imshow(img[Z, Y-64: Y+64, X-64: X+64])  # [zyx]
plt.imshow(img[Z, :, :]) 
plt.show()