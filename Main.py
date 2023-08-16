import time
from DataCollector.DataCollector import DataCollector
from Code import SleepStageDetect

# Subject Init
pass

# Sleep Staging Model Init
SleepStage = SleepStageDetect.SleepStageModel()

# Sleep Spindle Model Init
pass

# Channel Init
C3 = 14
M2 = 18
EOG = 31
EMG = 32
fs = 500
warming = 10

# Data Collector Init & Start
collector = DataCollector(C3, EOG, EMG, M2, fs, warming)
collector.dataCollect()

time.sleep(40)
now = time.time()

while True:
    if (time.time() - now >= 3):
        now = time.time()
        data = collector.dataGet(30)
        print(SleepStage.predict(data))
        print("Use Time: %.3fms" % ((time.time() - now) * 1000))
