import Config
import ModelLIB
import DataSet 
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, History
from keras.models import load_model
from HistoryGraph import draw_history_graph
import pickle
import os 
import csv
import pandas

from sklearn.metrics import classification_report, confusion_matrix
import ConfusionMatrix
import matplotlib.pyplot as plt 
from keras.preprocessing.image import ImageDataGenerator
import image_gen
import numpy as np
import PostProcessing

import cv2

import gc

from keras import backend as K

def Train_DataSet(TaskID, datasetBase, datasetConfig, modelConfig, train_data, valid_data, train_mask, valid_mask):
    
    Result = False
    #ROOT_DIR = os.getcwd()
    ROOT_DIR = datasetBase + "/LOGS"
    LOG_DIR = ROOT_DIR + "/logs"
    MODEL_DIR = ROOT_DIR + "/Models"
    
    LOG_DIR_FOLDER = LOG_DIR + "/Train_" + TaskID
    HistImgFile = "History_" + TaskID + ".png"
    HistDumpFile = "History_" + TaskID + ".txt"
    HistDumpFileCSV = "History_" + TaskID + ".csv"
    ModelFileName = modelConfig.MODEL_TYPE + "_" + TaskID + ".hdf5"
    Model_File_Format = ModelFileName + "_{epoch:03d}.hdf5"

    #config.LOG_PATH = LOG_DIR_FOLDER
    batch_size = modelConfig.BATCH_SIZE
  

    train_gen = image_gen.ImageDataGenerator(rotation_range=modelConfig.ROTATION_RANGE,
                                   width_shift_range=modelConfig.WIDTH_SHIFT_RANGE,
                                   height_shift_range=modelConfig.HEIGHT_SHIFT_RANGE,
                                   horizontal_flip = modelConfig.HORIZONTAL_FLIP,
                                   rescale=1.,
                                   zoom_range=0.2,
                                   fill_mode='nearest',
                                  cval=0)
    test_gen = image_gen.ImageDataGenerator(rescale=1.)

    if(len(train_data.shape) == 3):
        train_data = np.expand_dims(train_data, -1)  
    train_generator = train_gen.flow(train_data,train_mask,  batch_size)
    if(len(valid_data.shape) == 3):
        valid_data = np.expand_dims(valid_data, -1) 
    validation_generator = test_gen.flow(valid_data,valid_mask, batch_size)




    model = ModelLIB.Make_Model(modelConfig, datasetConfig)

    checkpointer = ModelCheckpoint(LOG_DIR_FOLDER + "/" + Model_File_Format, period=2)
    history = History()
    early_stopper = EarlyStopping(monitor='val_loss',patience=10,verbose=0)
    tensorboard = TensorBoard(log_dir=LOG_DIR_FOLDER, histogram_freq=0, write_graph=True, write_images=True)

    callbacks = []
    callbacks.append(checkpointer)
    callbacks.append(history)
    callbacks.append(early_stopper)
    callbacks.append(tensorboard)

    try :
        if(not os.path.isdir(LOG_DIR_FOLDER)):
            os.mkdir(LOG_DIR_FOLDER)
            
        #print(train_generator.shape, validation_generator.shape)
        history = model.fit_generator( train_generator, validation_data=validation_generator, 
                            steps_per_epoch=train_generator.length(),
                            validation_steps=validation_generator.length(), 
                            epochs=modelConfig.EPOCH,
                            workers = 0,
                            callbacks=callbacks)

        datasetConfig.SaveConfig(LOG_DIR_FOLDER)
        modelConfig.SaveConfig(LOG_DIR_FOLDER)
        model.save(MODEL_DIR + "/" + ModelFileName)

        draw_history_graph(history, LOG_DIR_FOLDER + "/" + HistImgFile)
        with open(LOG_DIR_FOLDER + "/" + HistDumpFile, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        pandas.DataFrame(history.history).to_csv(LOG_DIR_FOLDER + "/" + HistDumpFileCSV)
        save_TrainHistoryLog("Success", TaskID, ROOT_DIR, datasetConfig, modelConfig)
        Result = True


    except Exception as ex: 
        print(ex)
        if( len(os.listdir(LOG_DIR_FOLDER)) == 0):
            os.rmdir(LOG_DIR_FOLDER)
        save_TrainHistoryLog("Fail", TaskID, ROOT_DIR, datasetConfig, modelConfig)
        Result = False

    print("Train Done ! ")

    del model
    gc.collect()
    K.clear_session()
    return Result, ModelFileName



def Test_Dataset(TaskID, datasetBase, datasetConfig, ModelFileName, testData, testMask, filenames, modelInLogDir = False):
    
    #ROOT_DIR = os.getcwd()
    ROOT_DIR = datasetBase + "/LOGS"
    LOG_DIR = ROOT_DIR + "/logs"
    if(modelInLogDir):
        MODEL_DIR = LOG_DIR
    else:
        MODEL_DIR = ROOT_DIR + "/Models"

    LOG_DIR_FOLDER = LOG_DIR + "/Test_" + TaskID
    
    Mask_DIR = LOG_DIR_FOLDER + "/Mask"
    OVERLAY_DIR = LOG_DIR_FOLDER + "/OverLay"

    model = load_model(MODEL_DIR + "/" + ModelFileName)

    
    
    testData = np.asarray(testData)
    if(len(testData.shape) == 3):
        testData = np.expand_dims(testData, -1) 
    
    if(not os.path.isdir(LOG_DIR_FOLDER)):
        os.mkdir(LOG_DIR_FOLDER)

    if(not os.path.isdir(Mask_DIR)):
        os.mkdir(Mask_DIR)
    if(not os.path.isdir(OVERLAY_DIR)):
        os.mkdir(OVERLAY_DIR)
    try :    
        preds = model.predict(testData, batch_size = 1) 
        preds = preds > 0.5
        #print(preds.shape)
        processed = PostProcessing.PostProcessscing(preds)
        
        dice = None
        jaccard = None
        if (len(testMask) != 0):
            dices_ = Dice(processed, testMask)
            jacards_ = Jaccard_coef(processed, testMask)

            dice = dices_.mean()
            jaccard = jacards_.mean()
            print("dice ", dice, " jaccard ", jaccard)
        
        save_TestHistoryLog(TaskID, ROOT_DIR, ModelFileName, dice, jaccard, datasetConfig)
        

        for i in range(len(processed)):
            
            mask = processed[i,:,:,0]
            mask = mask * 255 
            cv2.imwrite(Mask_DIR + "/" + filenames[i] ,mask)

            fig = plt.figure()
            fig.set_size_inches(256/256, 1, forward=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            if (len(testMask) == 0):
                img = testData[i,:,:,0]
                plt.imshow(img, cmap = "gray")
            else:
                true_mask = testMask[i,:,:,0]
                true_mask = true_mask * 255
                plt.imshow(true_mask, cmap='gray')
            
            ax.imshow(mask, cmap='gray', alpha = 0.15, interpolation = 'nearest')
            plt.savefig(OVERLAY_DIR + "/" + filenames[i], dpi = 2048)  

            print(str(i), filenames[i])  

    except Exception as ex: 
        print(ex)
       
        os.rmdir(LOG_DIR_FOLDER)

    del model
    gc.collect()
    K.clear_session()
    print("Test Done ! ")
    return dice




def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    length = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    #print(y_pred.shape, y_true.shape)
    assert len(y_true.shape) == 4 and y_true.shape == y_pred.shape

    intersection = np.logical_and(y_true, y_pred)  
    true_sum = np.zeros(length)
    pred_sum = np.zeros(length)
    intersection_sum = np.zeros(length)
    for i in range(length):
        true_sum[i]= y_true[i,:,:,:].sum()
        pred_sum[i]= y_pred[i,:,:,:].sum()
        intersection_sum[i] = intersection[i,:,:,:].sum()

    dices = np.zeros(length)

    #for i in range(length):
    dices[:] = (2 * intersection_sum[:] + 1.) / (true_sum[:] + pred_sum[:] + 1.)

    return dices
    
def Jaccard_coef(y_true, y_pred):
    smooth = 1.
    length = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert len(y_true.shape) == 4 and y_true.shape == y_pred.shape

    intersection = np.logical_and(y_true, y_pred)  
    union = np.logical_or(y_true, y_pred) 
    union_sum = np.zeros(length)
    intersection_sum = np.zeros(length)
    for i in range(length):
        union_sum[i]= union[i,:,:,:].sum()
        intersection_sum[i] = intersection[i,:,:,:].sum()

    jacards = np.zeros(length)

    #for i in range(length):
    jacards[:] = (intersection_sum[:] + smooth) / (union_sum[:] + smooth)

    return jacards 
    
def save_TrainHistoryLog(arg, TaskID, RootDir,  datasetConfig, modelConfig):

    HistoryFile = RootDir + "/Segmentation_TrainHistory.csv"
    
    f = open(HistoryFile, 'a')
    f_writer = csv.writer(f)
    
    strLines = [TaskID]
    for a in dir(datasetConfig):
        if not a.startswith("__") and not callable(getattr(datasetConfig, a)):
            strLines.append("{}".format(getattr(datasetConfig, a)))
    for a in dir(modelConfig):
        if not a.startswith("__") and not callable(getattr(modelConfig, a)):
            strLines.append("{}".format(getattr(modelConfig, a)))

    strLines.append(arg)
    f_writer.writerow(strLines)
    f.close()

def save_TestHistoryLog(TaskID, RootDir, ModelFileName, dice, jacard,  datasetConfig):
    HistoryFile = RootDir + "/Segmentation_TestHistory.csv"
    
    f = open(HistoryFile, 'a')
    f_writer = csv.writer(f)
    
    strLines = [TaskID, ModelFileName, datasetConfig.DATASET, str(dice), str(jacard)]
    #strLines.append(datasetConfig.CLASSES)
    
    f_writer.writerow(strLines)
    f.close()

def get_Accucary(totalResult): 

    totalResult = np.asarray(totalResult)
    totalResult.flatten()
    
    
    
    accuracy = totalResult.sum() / len(totalResult)
    print(accuracy)
    return accuracy