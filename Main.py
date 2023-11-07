import time
from Utils.DataCollector.DataCollector import DataCollector
from Code.Utils import SleepStageDetect
from Utils.PortInput import StageInput, AudioInput, SlowOscillationInput
from Utils.SODetect import SlowOscillationDetector
from Subjects import Subject


# Channel Init
Amplifier = "BP"  # Amplifier device
use_channel = 2  # EEG, EOG, EMG
fs = 500  # Sampling Rate
warming = 10  # Buffer Length
preNight_path = None

# Staging Para
stagingInterval = 3  # the interval of SO detect, the unit is s
stagingDataLength = 30  # the data length when staging, the unit is s

# Subject Init
subject = Subject()
logDir = subject.get_fileName()

# SO Para Init
neg_thresh, pos_thresh, down_duration, up_duration = subject.subjectInit(preNight_path)
logPath = logDir + "/Subject_LOG.txt"
log = open(logPath, "a+")
wp = [0.5, 4]  # [low_freq_pass, high_freq_pass]
ws = [0.01, 8]  # [low_freq_stop, high_freq_stop]
filterOrder = 2  # the filter order
detectPauseInterval = 1  # the interval of SO detect while find the up state, the unit is s
detectInterval = 5  # the interval of SO detect, the unit is ms

soDataLength = 30  # the data length when so detecting, the unit is s
detectInterval = detectInterval / 1000
down_duration = down_duration / 1000
up_duration = up_duration / (2 * 1000)

if Amplifier == "ANT":
    C3 = 14  # C3 channel index in amplifier
    M2 = 18  # M2 channel index in amplifier
    EOG = 70  # EOG channel index in amplifier
    EMG = 71  # EMG channel index in amplifier
    EMGREF = -1  # EMGREF channel index in amplifier
elif Amplifier == "BP":
    C3 = 4  # C3 channel ind ex in amplifier
    M2 = 31  # M2 channel index in amplifier
    EOG = 57  # EOG channel index in amplifier
    EMG = 52  # EMG channel index in amplifier
    EMGREF = -1  # EMGREF channel index in amplifier
else:
    pass


# Sleep Staging Model Init
SleepStage = SleepStageDetect.SleepStageModel(useChannel=use_channel)

# Sleep Spindle Model Init
pass

# Slow Oscillation Model Init
SO_Dector = SlowOscillationDetector(wp, ws, fs, neg_thresh, pos_thresh, down_duration, up_duration, filterOrder)

# Port Writer
StagePort = StageInput(device=Amplifier)
# Cue Writer
SoundPort = AudioInput(device=Amplifier)
# SO Writer
SlowPort = SlowOscillationInput(device=Amplifier)

# Data Collector Init & Start
collector = DataCollector(C3, EOG, EMG, M2, EMGREF, use_channel, Amplifier, fs, warming)
collector.dataCollect()

stage = "W"  # Init Stage
stageCollector = SleepStageDetect.SleepStageCollector()
SO_enable = True

try:
    time.sleep(40)
    now_stage = time.time()
    now_soEnable = time.time()
    now_soDetect = time.time()

    while True:

        # Sleep Staging
        if (time.time() - now_stage >= stagingInterval):
            now_stage = time.time()
            data_Stage = collector.dataGet(stagingDataLength)
            stage, prob = SleepStage.predict(data_Stage)
            StagePort.writeStage(stage)
            count_str = stageCollector.update(stage)

            # Cue Play during TMR
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
        if (stageCollector.N2N3_count >= 3) and SO_enable and (time.time() - now_soDetect >= detectInterval):
            now_SO = time.time()
            data_SO = collector.dataGet_Single(soDataLength)

            if stageCollector.N2_count > 0:
                detectStage = "N2"
            else:
                detectStage = "N3"
            SO_state = SO_Dector.SO_find(data_SO, detectStage)

            if SO_state == "UpState":
                SO_enable = False
                now_soEnable = time.time()
                SlowPort.writeSO(SO_state, stage)

            now_soDetect = time.time()

        # Reset the SO Detect Access
        if not SO_enable:
            if time.time() - now_soEnable >= detectPauseInterval:
                SO_enable = True

        # Cue Play
        # if (time.time() - now_audio >= 5) and (time.time() - start_point <= 10 * 60):
        #     now_audio = time.time()
        #     soundPlay.audioPlay()
        #     print("Cue Use Time: %.3fms" % ((time.time() - now_audio) * 1000))

except KeyboardInterrupt:
    log.close()
