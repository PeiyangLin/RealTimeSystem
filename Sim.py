# Author: LangaLinn
# File Create: 2023/9/14 19:17

import numpy as np
import pandas as pd
import torch
import os
import time
from SleepStageDetect import SleepStageModel as SleepModel
from Simulation.Simuoutput import dataSim as Collector_Model
import warnings

warnings.filterwarnings("ignore")


def timecal(useTime):
    m, s = divmod(useTime, 60)
    h, m = divmod(m, 60)
    return "%02dh %02dm %.3fs" % (h, m, s)


now_init = time.time()

# Only Hyperparameters you will change:
Time_Sld = 30
C3 = 2
REog = 6
EMG = 7

basePath = "G:/DataSets/CIBR_data/"
database = "FL"
index = 3
database = basePath + database

# Channel init
fs = 500
label_stride = 30
model_stride = 30
yasa_stride = 30
realtime_stride = 3
warmingTime = 600

# Sleep Stage mapping
mapping_yasa = {"W": 0, "N1": 0, "N2": 2, "N3": 3, "R": 4}
mapping_model = {"W/N1": 0, "N2": 2, "N3": 3, "REM": 4}

# Staging model init
net = SleepModel()

# Paths init
if "FL" in database:
    dataPath = "%s/%03d/%03d_eeg_data.npy" % (database, index, index)
    LabelPath = "%s/%03d/hypno_30s.csv" % (database, index)
    P = "FL"
else:
    dataPath = "%s/sub%02d_night/sub%02d_night_eeg_data.npy" % (database, index, index)
    LabelPath = "%s/sub%02d_night/hypno_30s.csv" % (database, index)
    P = "IS"
SaveLabel = "../BenchMark/%s/%03d" % (P, index)
if not os.path.exists(SaveLabel):
    os.mkdir(SaveLabel)
print("Simulating Database %s, index %03d\n" % (P, index))

# Data Loading
print("Data loading...")

now = time.time()
data = np.load(dataPath)
label = pd.read_csv(LabelPath)
description = list(label["description"])
total_length = []
print("Done! Use time: %s\n" % timecal(time.time() - now))

# Ground Truth
print("Indexing ground truth...")
now = time.time()
GroundTruth = []
for d in description:
    if "2" in d:
        temp_stage = 2
    elif "3" in d:
        temp_stage = 3
    elif "R" in d:
        temp_stage = 4
    else:
        temp_stage = 0

    for _ in range(label_stride):
        GroundTruth.append(temp_stage)
GroundTruth = np.array(GroundTruth)
total_length.append(len(GroundTruth))
print("Done! Use time: %s\n" % timecal(time.time() - now))

# Realtime Predicted by Model
print("Predicting realtime data by our model...")
now_realtime = time.time()
Collector_model = Collector_Model(dataPath, C3=C3, REog=REog, EMG=EMG)
Collector_model.warmup(warmingTime)
realtime_model = []
usetime_model = []
for _ in range(warmingTime):
    realtime_model.append(0)
try:
    while True:
        now = time.time()
        dataout, (h, m, s) = Collector_model.dataout(Time_Sld, realtime_stride)

        pred, prob = net.predict_offline(dataout)

        useTime = float(time.time() - now)
        usetime_model.append(useTime)
        for _ in range(realtime_stride):
            realtime_model.append(mapping_model[pred])
        print("\r", pred, prob, "Time: %.3fms" % (useTime * 1000), "Sleep Time: %02d:%02d:%02d" % (h, m, s), end="")


except BaseException:
    realtime_model = np.array(realtime_model)
    usetime_model = np.array(usetime_model)
    print()
np.save(SaveLabel + "/Realtime_model.npy", realtime_model)
np.save(SaveLabel + "/Realtime_model_time.npy", usetime_model)
print("Done! Use time: %s\n" % timecal(time.time() - now_realtime))
