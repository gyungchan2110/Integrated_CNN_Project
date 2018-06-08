from Util import prepare_Task, TaskID_Generator
import os 
from Config import Config
import ClassificationTask 

if __name__ == "__main__":
    GPU_Num, BaseDataPath = prepare_Task("Classification")
    TaskID = TaskID_Generator() 
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_Num
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    Classes = []
    configs = []

    Classes.append("Normal")
    Classes.append("Abnormal")

    config = Config("Test_Experiment", (224,224,3), Classes, "Test")
    config.BATCH_SIZE = 64
    config.EPOCH = 1
    config.MODEL_TYPE = "RESNET152"

    configs.append(config)

    for i in range(len(configs)):
        model, filename = ClassificationTask.Train(TaskID, BaseDataPath, configs[i])
        ClassificationTask.Test(TaskID, BaseDataPath, configs[i], model, filename)