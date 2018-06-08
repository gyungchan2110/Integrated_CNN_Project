import Config
import ModelLIB_Diap as ModelLIB
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
import numpy as np

import gc

from keras import backend as K


def Train_DataSet(TaskID, datasetBase, datasetConfig, modelConfig, train_data, valid_data, train_labels, valid_labels):
    Result = False
    #ROOT_DIR = os.getcwd()
    ROOT_DIR = datasetBase + "/LOGS"
    LOG_DIR = ROOT_DIR + "/logs"
    MODEL_DIR = ROOT_DIR + "/Models"
    
    LOG_DIR_FOLDER = LOG_DIR + "/Train_" + TaskID
    print("Train_" + TaskID)
    HistImgFile = "History_" + TaskID + ".png"
    HistDumpFile = "History_" + TaskID + ".txt"
    HistDumpFileCSV = "History_" + TaskID + ".csv"
    ModelFileName = modelConfig.MODEL_TYPE + "_" + TaskID + ".hdf5"
    Model_File_Format = ModelFileName + "_{epoch:03d}.hdf5"

    #config.LOG_PATH = LOG_DIR_FOLDER
    batch_size = modelConfig.BATCH_SIZE
  

    train_gen = ImageDataGenerator(rotation_range=modelConfig.ROTATION_RANGE,
                                   width_shift_range=modelConfig.WIDTH_SHIFT_RANGE,
                                   height_shift_range=modelConfig.HEIGHT_SHIFT_RANGE,
                                   horizontal_flip = modelConfig.HORIZONTAL_FLIP,
                                   rescale=1.,
                                   zoom_range=0.2,
                                   fill_mode='nearest',
                                  cval=0)
    test_gen = ImageDataGenerator(rescale=1.)
    if(len(train_data.shape) == 3):
        train_data = np.expand_dims(train_data, -1)  
    train_generator = train_gen.flow(train_data,train_labels,  batch_size)
    if(len(valid_data.shape) == 3):
        valid_data = np.expand_dims(valid_data, -1) 
    validation_generator = test_gen.flow(valid_data,valid_labels, batch_size)

    model = ModelLIB.Make_Model(modelConfig, datasetConfig)

    checkpointer = ModelCheckpoint(LOG_DIR_FOLDER + "/" + Model_File_Format, period=2)
    history = History()
    early_stopper = EarlyStopping(monitor='val_loss',patience=10,verbose=0)
    tensorboard = TensorBoard(log_dir=LOG_DIR_FOLDER, histogram_freq=0, write_graph=True, write_images=True)

    callbacks = []
    callbacks.append(checkpointer)
    callbacks.append(history)
    #callbacks.append(early_stopper)
    callbacks.append(tensorboard)

    try :
        if(not os.path.isdir(LOG_DIR_FOLDER)):
            os.mkdir(LOG_DIR_FOLDER)
            
        print(len(train_generator), len(validation_generator))
        history = model.fit_generator( train_generator, validation_data=validation_generator, 
                            steps_per_epoch=len(train_generator),
                            validation_steps=len(validation_generator), 
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



def Test_Dataset(TaskID, datasetBase, datasetConfig, ModelFileName, testData, true_labels, modelInLogDir = False):
    
    #ROOT_DIR = os.getcwd()
    ROOT_DIR = datasetBase + "/LOGS"
    LOG_DIR = ROOT_DIR + "/logs"
    if(modelInLogDir):
        MODEL_DIR = LOG_DIR
    else:
        MODEL_DIR = ROOT_DIR + "/Models"
    
    LOG_DIR_FOLDER = LOG_DIR + "/Test_" + TaskID
    print("Test_" + TaskID)
    HistDumpFile = "History_" + TaskID + ".txt"
    ConfusionMatrixImg = "Confusion_" + TaskID + ".png"
    ResultFile = "Result_" + TaskID + ".csv"
    ReportFile = "Report" + TaskID + ".csv"
    
    model = load_model(MODEL_DIR + "/" + ModelFileName)
    
    testData = np.asarray(testData)
    if(len(testData.shape) == 3):
        testData = np.expand_dims(testData, -1) 

    total_result = []
    total_result_label = []
    total_true_label = []

    num_classes = datasetConfig.NUM_CLASS
    classes = datasetConfig.CLASSES
    
    if(not os.path.isdir(LOG_DIR_FOLDER)):
        os.mkdir(LOG_DIR_FOLDER)

    preds = model.predict(testData) 
    pred_result = preds.argmax(axis = 1)
    result = np.zeros(pred_result.shape)
    #print(true_Label)
    #print(pred_result)
    
    for i in range(len(result)):
        result[i] = true_labels[i][pred_result[i]]
    true_labels = np.asarray(true_labels)
    true_labels_binary = true_labels.argmax(axis = 1)
    print(result)
    accuracy = result.sum() / len(result)
    print(accuracy)

    # accuracy = get_Accucary(total_result.flatten())
    con_matrix = confusion_matrix(true_labels_binary, pred_result)
    ConfusionMatrix.plot_confusion_matrix(con_matrix, classes, saveImgFile = True, ImgFilename =LOG_DIR_FOLDER + "/" +ConfusionMatrixImg)           
    report = classification_report(true_labels_binary, pred_result, target_names=classes)

    save_TestHistoryLog(TaskID, ROOT_DIR, ModelFileName, accuracy, datasetConfig)
    save_TestResultLog(TaskID, ROOT_DIR, result, ModelFileName,datasetConfig.DATASET)

    f = open(LOG_DIR_FOLDER + "/" + ReportFile, 'w')
    f.write(report)
    #f.write(np.array2string(total_result))
    f.close()
    
    del model

    gc.collect()
    K.clear_session()
    print("Test Done ! ")
    return accuracy

def save_TrainHistoryLog(arg, TaskID, RootDir,  datasetConfig, modelConfig):

    HistoryFile = RootDir + "/Classification_TrainHistory.csv"
    
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

def save_TestHistoryLog(TaskID, RootDir, ModelFileName, accuracy,  datasetConfig):
    HistoryFile = RootDir + "/Classification_TestHistory.csv"
    
    f = open(HistoryFile, 'a')
    f_writer = csv.writer(f)
    
    strLines = [TaskID, ModelFileName, datasetConfig.DATASET, str(accuracy), datasetConfig.NUM_CLASS]
    strLines.append(datasetConfig.CLASSES)
    
    f_writer.writerow(strLines)
    f.close()

def save_TestResultLog(TaskID, RootDir, Result,  ModelFile, dataSet):
    HistoryFile = RootDir + "/" + dataSet + "_Result.csv"
    
    f = open(HistoryFile, 'a')
    f_writer = csv.writer(f)
    
    strLines = [ModelFile]
    Result = np.asarray(Result, dtype = "uint8")
    strLines.append(Result.tolist())
    
    f_writer.writerow(strLines)
    f.close()
    
def get_Accucary(totalResult): 
    totalsum = 0
    totallen = 0
    totalResult = np.asarray(totalResult)
    totalResult.flatten()
    
    
    
    accuracy = totalResult.sum() / len(totalResult)
    print(accuracy)
    return accuracy