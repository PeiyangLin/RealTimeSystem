import numpy as np
import torch
import time
import threading
from DataCollector.DataCollector import DataCollector

# Subject Init


# Channel Init
C3 = 14
C4 = 16
M2 = 18
EOG = 31
fs = 500
warming = 10

# Data Collector Init & Start
collector = DataCollector(C3, EOG, M2, fs, warming)
collector.dataCollect()



sleep = 3
while True:
    time.sleep(sleep)
    temp = collector.dataGet(sleep)

    temp = np.array(temp)
    print(temp.shape)

