import numpy as np
import torch
import time
import threading
from DataCollector.DataCollector import DataCollector

# Subject Init
pass

# Sleep Staging Model Init
pass

# Sleep Spindle Model Init
pass

# Channel Init
C3 = 14
C4 = 16
M2 = 18
EOG = 31
fs = 500
warming = 0.5

# Data Collector Init & Start
collector = DataCollector(C3, EOG, M2, fs, warming)
collector.dataCollect()

while True:
    time.sleep(5)
    data = collector.dataGet_All()
    print(data.shape)
