# In[]
import cv2  
import numpy as np
import matplotlib.pyplot as plt 

ImgPath = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180327_151800_2Classes_Original/train/Normal/Img_20180130_170115.png"
MaskPath = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180327_151800_2Classes_Original_LungMask/train/Normal/Img_20180130_170115.png"

Img = cv2.imread(ImgPath, 0)
Mask = cv2.imread(MaskPath, 0)
Img = cv2.resize(Img, (1024,1024))
Img = np.asarray(Img)
Mask = np.asarray(Mask)

Image = np.stack((Img, Img, Mask), -1)

print(Image.shape)
plt.imshow(Image)
plt.show()


