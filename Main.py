import time
from DataCollector.DataCollector import DataCollector
from Code import SleepStageDetect
from PortInput import StageInput, AudioInput

# Subject Init
pass

# Sleep Staging Model Init
SleepStage = SleepStageDetect.SleepStageModel()

# Sleep Spindle Model Init
pass

# Port Writer
portIO = StageInput()
# Audio Writer
soundPlay = AudioInput()

# Channel Init
C3 = 14
M2 = 18
EOG = 70
EMG = 71
fs = 500
warming = 30

# Data Collector Init & Start
collector = DataCollector(C3, EOG, EMG, M2, fs, warming)
collector.dataCollect()

time.sleep(40)
start_point = time.time()
now_stage = time.time()
now_audio = time.time()

while True:
    if (time.time() - now_stage >= 5):
        now_stage = time.time()
        data = collector.dataGet(30)
        stage, prob = SleepStage.predict(data)
        portIO.writeStage(stage)
        if (stage == "N2" or stage == "N3") and (time.time() - start_point >= 10 * 60):
            soundPlay.audioPlay()
        print(stage, prob)
        print("Staging Use Time: %.3fms" % ((time.time() - now_stage) * 1000))

    if (time.time() - now_audio >= 5) and (time.time() - start_point <= 10 * 60):
        now_audio = time.time()
        soundPlay.audioPlay()
        print("Audio Use Time: %.3fms" % ((time.time() - now_audio) * 1000))
