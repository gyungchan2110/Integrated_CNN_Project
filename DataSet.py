#-*- coding: utf-8 -*-

import numpy as np
import glob
import os
#from processor import process_image
from keras.preprocessing.image import ImageDataGenerator
#from image_gen import ImageDataGenerator
import cv2
from skimage import transform, io, img_as_float, exposure
import Config
import time
import matplotlib.pyplot as plt
import csv 

def get_Data_Set(datasetConfig, datasetBase, dataType):
    start_time = time.time()
    filename_arrs = []
    #print(classes)
    classes = datasetConfig.CLASSES
    path = datasetBase + "/" + datasetConfig.DATASET + "/" + dataType
    
    true_labels = []
    image_arr = []
    
    
    ImgReadFlag = cv2.IMREAD_GRAYSCALE  
    if(datasetConfig.IMG_SHAPE[2] == 3):
        ImgReadFlag = cv2.IMREAD_COLOR
    
    print(path)
    files = glob.glob(path + "/*/*.png")
    
    for file in files:
        #print(file)
        filename_arrs.append(file)
        image_data = cv2.imread(file, ImgReadFlag)
        image_data = np.asarray(image_data, dtype = "float")
        image_data = cv2.resize(image_data, (datasetConfig.IMG_SHAPE[0], datasetConfig.IMG_SHAPE[1]))
        if(len(image_data.shape) != 3):
            image_data = np.expand_dims(image_data, axis = -1)
        image_data /= 255.
        image_data = np.stack(image_data, axis=0)
        image_arr.append(image_data)
        #print(image_data.shape)
        
        file_Class = file.split('/')[-2]
        classes = np.asarray(classes)
        index = np.argwhere(classes == file_Class)
        label = np.zeros(len(classes))
        label[index] = 1
        true_labels.append(label)
        
    image_arr = np.asarray(image_arr)
    mean =  image_arr.flatten().mean()
    std =  image_arr.flatten().std()
    print(dataType , len(files))
    print(" : --- %s seconds ---" %(time.time() - start_time))
    return image_arr,filename_arrs, true_labels, mean ,std

def get_Data_Set_Level(datasetConfig, datasetBase, dataType):
    start_time = time.time()
    filename_arrs = []
    #print(classes)
    classes = datasetConfig.CLASSES
    path = datasetBase + "/" + datasetConfig.DATASET + "/" + dataType
    
    true_labels = []
    image_arr = []
    
    csvFile = datasetBase + "/" + datasetConfig.DATASET + "/xLevel.csv"
    
    ImgReadFlag = cv2.IMREAD_GRAYSCALE  
    if(datasetConfig.IMG_SHAPE[2] == 3):
        ImgReadFlag = cv2.IMREAD_COLOR
    
    print(path)
    files = glob.glob(path + "/*.png")
    
    for file in files:
        #print(file)
        filename_arrs.append(file)
        image_data = cv2.imread(file, ImgReadFlag)
        image_data = np.asarray(image_data, dtype = "float")
        image_data = cv2.resize(image_data, (datasetConfig.IMG_SHAPE[0], datasetConfig.IMG_SHAPE[1]))
        if(len(image_data.shape) != 3):
            image_data = np.expand_dims(image_data, axis = -1)
        image_data /= 255.
        image_data = np.stack(image_data, axis=0)
        image_arr.append(image_data)
        #print(image_data.shape)
        
        file_Class = file.split('/')[-2]
        classes = np.asarray(classes)
        index = np.argwhere(classes == file_Class)
        label = np.zeros(len(classes))
        vl = 0
        f = open(csvFile, 'r', encoding = "utf-8", newline='')
        csvReader = csv.reader(f)
        for row in csvReader:
            if(row[0] == file):
                vl = float(row[1])
                break
        label[index] = vl
        true_labels.append(label)
        
    image_arr = np.asarray(image_arr)
    mean =  image_arr.flatten().mean()
    std =  image_arr.flatten().std()
    print(dataType , len(files))
    print(" : --- %s seconds ---" %(time.time() - start_time))
    return image_arr,filename_arrs, true_labels, mean ,std




def get_Data_Set_Segmentaion(datasetConfig, datasetBase, dataType):
    start_time = time.time()


    Imgs = []
    Masks = []
    filenames = []

    ImgPath = datasetBase + "/" + datasetConfig.DATASET + "/Imgs/" + dataType
    MaskPath = datasetBase + "/" + datasetConfig.DATASET + "/Masks/" + dataType

   
    isMaskPath = os.path.isdir(MaskPath)

    print(ImgPath)
    files = glob.glob(ImgPath + "/*.png")


    for file in files:
        
        filename = file.split("/")[-1]

        filenames.append(filename)
        
        image_data = cv2.imread(file, 0)
        image_data = np.asarray(image_data, dtype = "float")
        image_data = cv2.resize(image_data, (datasetConfig.IMG_SHAPE[0], datasetConfig.IMG_SHAPE[1]))
        #print(image_data.shape)
        
        
        
        if(len(image_data.shape) !=3 ):
            image_data = np.expand_dims(image_data, axis = -1)
            
        #image_data /= 255.
        image_data = np.stack(image_data, axis=0)
        Imgs.append(image_data)

        if(isMaskPath):
            mask = cv2.imread(MaskPath + "/" + filename)
            mask = cv2.resize(mask, (datasetConfig.IMG_SHAPE[0], datasetConfig.IMG_SHAPE[1]))
            mask = mask[:,:,0]
            mask = np.asarray(mask, dtype = "uint8")
            
            ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            

            mask = mask > 127 

            mask = np.asarray(mask, dtype = "uint8")
            if(len(mask.shape) !=3 ):
                mask = np.expand_dims(mask, axis = -1)
            mask = np.stack(mask, axis=0)
            Masks.append(mask)

    Imgs = np.asarray(Imgs)
    Masks = np.asarray(Masks)
    mean = Imgs.mean()
    std = Imgs.std()
    # Imgs -= mean
    # Imgs /= Imgs.std()

    print(dataType , len(files))
    print(" : --- %s seconds ---" %(time.time() - start_time))


    return Imgs, filenames,Masks, mean,std



def get_Data_Generator(config, dataSetBase, dataType):
    PATH = dataSetBase + "/" + config.DATASET + "/" + dataType
    datagen = ImageDataGenerator(
        #rescale=1./config.RESCALE_RATE,
        shear_range=config.SHEAR_RANGE,
        horizontal_flip = config.HORIZONTAL_FLIP,
        rotation_range = config.ROTATION_RANGE,
        width_shift_range = config.WIDTH_SHIFT_RANGE,
        height_shift_range = config.HEIGHT_SHIFT_RANGE,
        featurewise_center=True,
        featurewise_std_normalization=True
    )
    generator = datagen.flow_from_directory(
        PATH,
        target_size = (config.IMG_SHAPE[0], config.IMG_SHAPE[1]),
        batch_size = config.BATCH_SIZE,
        classes = config.CLASSES,
        class_mode='categorical')


    return generator

def get_Data_Generator_withdout_aug(config, dataSetBase, dataType):
    PATH = dataSetBase + "/" + config.DATASET + "/" + dataType
    
    datagen = ImageDataGenerator(
        #rescale=1./config.RESCALE_RATE,
        featurewise_center=True,
        featurewise_std_normalization=True
    )
    generator = datagen.flow_from_directory(
        PATH,
        target_size=(config.IMG_SHAPE[0], config.IMG_SHAPE[1]),
        batch_size=config.BATCH_SIZE,
        classes=config.CLASSES,
        class_mode='categorical')

    return generator

def get_Data_Set_Classification(ImgPath, img_shape, batch_size = 8):

    Imgs = []
    Masks = []
    filenames = []

    for i, filename in enumerate(os.listdir(ImgPath)):
        filetitle = filename.split(".")

        img = io.imread(os.path.join(ImgPath, filename))
        img = transform.resize(img, img_shape)
        img = np.expand_dims(img, -1)

        Imgs.append(img)

        filenames.append(filename)
    Imgs = np.array(Imgs)
    Imgs -= Imgs.mean()
    Imgs /= Imgs.std()

    return Imgs, Masks, filenames


def get_Data_Set_Segmentaion_Cardiac(ImgPath, MaskPath, batch_size = 8, img_shape = (256,256)):
    
    Imgs = []
    Masks = []

    for i, filename in enumerate(os.listdir(ImgPath)):
        filetitle = filename.split(".")
        maskFileName = filetitle + "_0.tif"
        img = io.imread(os.path.join(ImgPath, filename))
        img = transform.resize(img, img_shape)
        img = np.expand_dims(img, -1)

        mask = io.imread(os.path.join(MaskPath, maskFileName))
        mask = transform.resize(mask, img_shape)
        mask = np.expand_dims(mask, -1)

        Imgs.append(img)
        Masks.append(mask)

    Imgs = np.array(Imgs)
    Masks = np.array(Masks)

    Imgs -= Imgs.mean()
    Imgs /= Imgs.std()

    return Imgs, Masks

def get_Data_Set_Segmentation_Test(ImgPath, img_shape):
    
    Imgs = []
    FileNames = []

    for i, filename in enumerate(os.listdir(ImgPath)):
        if(os.path.isdir(os.path.join(ImgPath, filename))):
            continue
        img = img_as_float(io.imread(os.path.join(ImgPath, filename)))
        img = transform.resize(img, img_shape)
        img = np.expand_dims(img, -1)

        Imgs.append(img)
        FileNames.append(filename)

    Imgs = np.array(Imgs)
    FileNames = np.array(FileNames)

    Imgs -= Imgs.mean()
    Imgs /= Imgs.std()

    return Imgs, FileNames
