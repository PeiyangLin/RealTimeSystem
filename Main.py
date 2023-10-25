import time
from DataCollector.DataCollector import DataCollector_EEG, DataCollector_EOG, DataCollector_ALL
import SleepStageDetect
from PortInput import StageInput, AudioInput, SlowOscillationInput
from SODetect import SlowOscillationDetector
from Subjects import Subject
import os

# Channel Init
Amplifier = "ANT"  # Amplifier device
use_channel = 2  # EEG, EOG, EMG
fs = 500  # Sampling Rate
warming = 5  # Buffer Length

# Staging Para
stagingInterval = 3  # the interval of SO detect, the unit is s
stagingDataLength = 30  # the data length when staging, the unit is s

# SO Para Init
wp = [0.5, 4]  # [low_freq_pass, high_freq_pass]
ws = [0.01, 8]  # [low_freq_stop, high_freq_stop]
neg_thresh = -40  # negative threshold
pos_thresh = 15  # positive threshold
filterOrder = 2  # the filter order
detectPauseInterval = 1  # the interval of SO detect while find the up state, the unit is s
detectInterval = 5  # the interval of SO detect, the unit is ms
down_duration = 700
up_duration = 700

soDataLength = 30  # the data length when so detecting, the unit is s
detectInterval = detectInterval / 1000
down_duration = down_duration / 1000
up_duration = up_duration / (2 * 1000)

if Amplifier == "ANT":
    C3 = 14  # C3 channel index in amplifier
    M2 = 18  # M2 channel index in amplifier
    EOG = 70  # EOG channel index in amplifier
    EMG = 71  # EMG channel index in amplifier
    EMGREF = 0  # EMGREF channel index in amplifier
elif Amplifier == "BP":
    C3 = 4  # C3 channel index in amplifier
    M2 = 31  # M2 channel index in amplifier
    EOG = 57  # EOG channel index in amplifier
    EMG = 52  # EMG channel index in amplifier
    EMGREF = 0  # EMGREF channel index in amplifier
else:
    pass

# Subject Init
subject = Subject()
logDir = "LOG/%s" % subject.get_fileName()
logPath = logDir + "/Subject_LOG.txt"
isExists = os.path.exists(logDir)

if not isExists:
    os.mkdir(logDir)
    os.mkdir(logDir + "")
    os.mknod(logPath)
log = open(logPath, "a+")
log_info = subject.info
for key in log_info:
    log.write("%s: %s\n" % (key, log_info[key]))

# Sleep Staging Model Init
SleepStage = SleepStageDetect.SleepStageModel(useChannel=use_channel)

# Sleep Spindle Model Init
pass

# Slow Oscillation Model Init
SO_Dector = SlowOscillationDetector(wp, ws, fs, neg_thresh, pos_thresh, down_duration, up_duration, filterOrder)

# Port Writer
StagePort = StageInput(device=Amplifier)
# Audio Writer
SoundPort = AudioInput(device=Amplifier)
# SO Writer
SlowPort = SlowOscillationInput(device=Amplifier)

# Data Collector Init & Start
if use_channel == 1:
    collector = DataCollector_EEG(C3, M2, fs, warming)
elif use_channel == 2:
    collector = DataCollector_EOG(C3, EOG, M2, fs, warming)
elif use_channel == 3:
    collector = DataCollector_ALL(C3, EOG, EMG, M2, fs, warming)
collector.dataCollect()

stage = "W"  # Init Stage
N2_count = 0  # Init N2 Stage Count
N3_count = 0  # Init N3 Stage Count
R_count = 0  # Init REM Stage Count
N2N3_count = 0
SO_enable = True

try:
    time.sleep(40)
    start_point = time.time()
    now_stage = time.time()
    now_audio = time.time()
    now_soEnable = time.time()
    now_soDetect = time.time()

    while True:

        # Sleep Staging
        if (time.time() - now_stage >= stagingInterval):
            now_stage = time.time()
            data_Stage = collector.dataGet(stagingDataLength)
            stage, prob = SleepStage.predict(data_Stage)
            StagePort.writeStage(stage)
            count_str = "No Counting"

            if stage == "N2":
                N2_count += 1
                N2N3_count += 1
                count_str = "N2 Count: %d" % N2_count
            else:
                N2_count = 0

            if stage == "N3":
                N3_count += 1
                N2N3_count += 1
                count_str = "N3 Count: %d" % N3_count
            else:
                N3_count = 0

            if stage == "R":
                R_count += 1
                N2N3_count = 0
                count_str = "REM Count: %d" % R_count
            else:
                R_count = 0

            if stage == "W/N1":
                N2N3_count = 0

            # Audio Play during TMR
            pass

            outputString = ("Stage: %s, Staging Prob: %s, StagingCount: %s, %s, %s\n" %
                            (str(stage),
                             str(prob),
                             count_str,
                             "Staging Use Time: %.3fms" % ((time.time() - now_stage) * 1000),
                             time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
            print(outputString)
            log.write(outputString)

        # SO Detect
        if (N2N3_count >= 3) and SO_enable and (time.time() - now_soDetect >= detectInterval):
            now_SO = time.time()
            data_SO = collector.dataGet_Single(soDataLength)

            if N2_count > 0:
                detectStage = "N2"
            else:
                detectStage = "N3"
            SO_state = SO_Dector.SO_find(data_SO, detectStage)
            SlowPort.writeSO(SO_state)
            # print(SO_state)

            if SO_state == "UpState":
                SO_enable = False
                now_soEnable = time.time()
                print("UP State, Use Time: %.3fms" % ((now_soEnable - now_SO) * 1000))

            now_soDetect = time.time()

            # print("SO time: %.3f" % ((now_soDetect-now_SO) * 1000))

        # Reset the SO Detect Access
        if not SO_enable:
            if time.time() - now_soEnable >= detectPauseInterval:
                SO_enable = True

        # Audio Play
        # if (time.time() - now_audio >= 5) and (time.time() - start_point <= 10 * 60):
        #     now_audio = time.time()
        #     soundPlay.audioPlay()
        #     print("Audio Use Time: %.3fms" % ((time.time() - now_audio) * 1000))

except KeyboardInterrupt:
    log.close()
