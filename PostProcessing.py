
import cv2
import numpy as np
import matplotlib.pyplot as plt

def PostProcessscing(mask):
    
   # for i, filename in enumerate(os.listdir(DetectedMasksPath) ):
    #print(filename)
    assert len(mask.shape) == 4
    imgCnt = len(mask)
    #imgs = mask
    #imgs = np.asarray(imgs, dtype = "float")
    
    mask = np.asarray(mask, dtype = "uint8")
    imgs = np.copy(mask)
    imgs = imgs * 255
    kernel = np.ones((5,5),np.uint8)

    ProcessedImgs = []
    for i in range(imgCnt) : 
        img = imgs[i,:,:,:]
        erosion = cv2.erode(img,kernel,iterations = 1)
        dilation = cv2.dilate(erosion,kernel,iterations = 1)

        dilationImg = np.copy(dilation)

        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(dilationImg, 127, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE )
        
        areas = []
        for cnt in contours:
            areas.append(cv2.contourArea(cnt))

        #print(len(areas))
        
        if(len(areas) == 0):
            continue
        areas = np.array(areas)
        areas_cp = np.copy(areas)

        maxindex = np.argmax(areas)
        areas_cp[maxindex] = 0
        secondmaxindex = np.argmax(areas_cp)  

        for i, cnt in enumerate(contours):
            if (i != maxindex) and (i != secondmaxindex) :
                cv2.drawContours(dilation, contours, i, color = (0,0,0), thickness = -1 )


        erosion = cv2.erode(dilation,kernel,iterations = 1)
        img_Post = cv2.dilate(erosion,kernel,iterations = 1)

        if(len(img_Post.shape) < 3):
            img_Post = np.expand_dims(img_Post, -1)
        img_Post = np.stack(img_Post, axis=0)
        ProcessedImgs.append(img_Post)
    #ProcessedImgs = np.asarray(imgs, dtype = "float")
    ProcessedImgs = np.asarray(ProcessedImgs, dtype = "uint8")
    ProcessedImgs = ProcessedImgs / 255
    
    #print(ProcessedImgs.shape)
    return ProcessedImgs