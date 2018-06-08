import socket 

from datetime import datetime

def prepare_Task(taskType) : 
    

    GPU_ServerType = "Local_83"

    hostName = socket.gethostname()
    if(hostName == "deepserver"):
        GPU_ServerType = "Local_83"
    elif (hostName[:3] == "csi"):
        GPU_ServerType = "Kakao"
    elif (hostName == "user-All-Series"):
        GPU_ServerType = "Local_180"    
    elif (hostName == "User-PC"):
        GPU_ServerType = "Local_PC" 

    BaseDataSetPath = ""
    GPU_Num = ""

    if(GPU_ServerType == "Local_83"):
        BaseDataSetPath = "/data/gyungchan2110/" + taskType
        GPU_Num = "3"
    elif (GPU_ServerType == "Kakao"):
        BaseDataSetPath = "/data/private/" + taskType
        GPU_Num = "0"
    elif (GPU_ServerType == "Local_180"):
        BaseDataSetPath = "/data6/gyungchan2110/" + taskType
        GPU_Num = "2"
    elif (GPU_ServerType == "Local_PC"):
        BaseDataSetPath = ""
        GPU_Num = "0"

    return GPU_Num, BaseDataSetPath



def TaskID_Generator():
    currentTime = datetime.now()
    strTime = "%04d%02d%02d_%02d%02d%02d" %(currentTime.year, currentTime.month, currentTime.day,currentTime.hour, currentTime.minute, currentTime.second)
    return strTime

#def TaskID_FromModelFileName(modelFileName):
    