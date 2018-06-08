# -*- coding: utf-8 -*- 

import math
import numpy as np
import os 



class Config(object):

    ConfigType = None
    NAME = ""
    
    
    def __init__(self, Name, Img_Shape, Classes, Dataset):
        pass


    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def SaveConfig(self, LogPath):

        if(not os.path.isdir(LogPath)): 
            os.mkdir(LogPath)

        if(not os.path.isdir(LogPath)): 
            print("Error : Log Path is Empty!")
            return 
        
        fileName = self.NAME + "_" +  self.ConfigType + ".txt"    
        filePath = LogPath + "/" + fileName
    
        f = open(filePath, 'a', encoding='utf-8')
        
        
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                f.writelines("{:30} {}\n".format(a, getattr(self, a)))
        f.close()

class DataSetConfig(Config):
    ConfigType = "DataSet"
    NAME = ""  
    NUM_CLASS = 0
    IMG_SHAPE = (224,224,1)
    CLASSES = []
    DATASET = ""

    def __init__(self, Name, Img_Shape, Classes, Dataset):
        self.NAME = Name
        self.IMG_SHAPE = Img_Shape
        self.CLASSES = Classes
        self.NUM_CLASS = len(self.CLASSES)
        self.DATASET = Dataset
    
    
    
class ModelConfig(Config):    
    ConfigType = "Model"
    
    NAME = ""

    MODEL_TYPE = "VGG16"
    OPTIMIZER = "ADAM"

    BATCH_SIZE = 64
    LEARNING_RATE = 1e-5
    DECAY = 1e-6
    MOMENTUM = 0.9

    EPOCH = 200 

    PRETRAINED_MODEL = None
    LOSS = "categorical_crossentropy"

    ## Augmentation 

    RESCALE_RATE = 255
    SHEAR_RANGE = 0
    HORIZONTAL_FLIP = False
    ROTATION_RANGE = 7.0
    WIDTH_SHIFT_RANGE = 0
    HEIGHT_SHIFT_RANGE = 0

    def __init__(self, Name, Model_type, Optimizer,Batch_Size, Learning_rate = 1e-5):
        self.NAME = Name
        self.MODEL_TYPE = Model_type
        self.OPTIMIZER = Optimizer
        self.LEARNING_RATE = Learning_rate
        self.BATCH_SIZE = Batch_Size
