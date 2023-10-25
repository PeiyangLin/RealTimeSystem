# Author: LangaLinn
# File Create: 2023/10/7 14:33

import numpy as np
import pandas as pd
import mne
import torch
import os
import time
from tqdm import tqdm
from SleepStageDetect import SleepStageModel as SleepModel
# from DataProcess import dataProcessor
from Simulation.Simuout import dataSim as Collector_Model
import warnings
from scipy import signal

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# Function Init
# def timecal(useTime):
#     m, s = divmod(useTime, 60)
#     h, m = divmod(m, 60)
#     return "%02dh %02dm %.3fs" % (h, m, s)

def timecal(useTime):
    ms = useTime - int(useTime)
    useTime = int(useTime)
    m, s = divmod(useTime, 60)
    h, m = divmod(m, 60)
    s = s + ms
    return "%02dh %02dm %.3fs" % (h, m, s)


def Filter_init(l_freq_p=0.5, h_freq_p=4, l_freq_s=0.1, h_freq_s=5, fs=500, order=2):
    LowPassFre = h_freq_p
    HighPassFre = l_freq_p

    h_freq_p = h_freq_p * 2 / fs
    l_freq_p = l_freq_p * 2 / fs
    wp = [l_freq_p, h_freq_p]

    h_freq_s = h_freq_s * 2 / fs
    l_freq_s = l_freq_s * 2 / fs
    ws = [l_freq_s, h_freq_s]

    N, wn = signal.buttord(wp, ws, 5, 40)
    N = order

    b, a = signal.butter(N, wn, "bandpass")

    w, gd = signal.group_delay((b, a), w=500, fs=500)

    Filter_Delay = np.mean(gd[np.min(np.where(w > HighPassFre)): np.max(np.where(w < LowPassFre))])

    Filter_Delay = round(Filter_Delay) // 2

    delay = np.zeros(Filter_Delay, dtype=np.float64)

    return b, a, N, Filter_Delay, delay


def SO_filter_filtfilt(b, a, x):
    x_flip = np.flip(x)
    x_long = np.concatenate([x, x_flip])
    y = signal.filtfilt(b, a, x_long)
    y = y[:len(x)]
    return y


def SO_filter_seq(b, a, x, fs=500):
    y = []
    total_length = len(x) // fs

    y0, zf = signal.lfilter(b, a, x[:fs], zi=np.zeros(len(b) - 1))

    y += list(y0)

    for i in range(1, total_length):
        yi, zf_new = signal.lfilter(b, a, x[i * fs: (i + 1) * fs], zi=zf)
        zf = zf_new
        y += list(yi)

    y = np.array(y)
    return y


def SO_filter(b, a, x, delay):
    x = np.concatenate([x, delay])
    y = signal.lfilter(b, a, x)
    y = y[Filter_Delay:]
    return y


# Only Hyperparameters you will change:
SaveLabel = "F:/35-2023_10_19-HongJiakang-Female"
dataPath = SaveLabel + "/Sleep.vhdr"

warmingTime = 300
Time_Sld = 30
Amplifier = "BP"
useChannel = 2
fs = 500
realtime_stride = 3  # The unit is sec

so_stride = 4  # The unit is msec
neg_thresh = -40  # negative threshold
pos_thresh = 15  # positive threshold
filterOrder = 2  # the filter order
detectPauseInterval = 1  # the interval of SO detect while find the up state, the unit is s

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

# Sleep Stage mapping
mapping_model = {"W/N1": 0, "N2": 2, "N3": 3, "R": 4}

# Data Loading
print("Data loading...")

now = time.time()
raw = mne.io.read_raw_brainvision(dataPath, preload=True)
raw.filter(0.1, 40)
data, _ = raw[:]
print("Done! Use time: %s\n" % timecal(time.time() - now))

# Realtime Predicted by Model
print("Predicting realtime data by our model...")
now_realtime = time.time()
Collector_model = Collector_Model(data=data, C3=C3, REog=EOG, M2=M2)
Collector_model.warmup(warmingTime)
realtime_model = []
usetime_model = []
so_onset = []
so_duration = []
so_description = []

staging_onset = []
staging_duration = []
staging_description = []
for _ in range(warmingTime):
    realtime_model.append(0)

net = SleepModel(useChannel=useChannel)

N2N3_Count = 0
b, a, N, Filter_Delay, delay = Filter_init(0.5, 4, 0.1, 5, fs, 2)

try:
    while True:
        now = time.time()
        dataout, (h, m, s) = Collector_model.dataout(Time_Sld, realtime_stride)

        pred, prob = net.predict_offline(dataout)

        useTime = float(time.time() - now)
        usetime_model.append(useTime)
        for _ in range(realtime_stride):
            realtime_model.append(mapping_model[pred])
        staging_onset.append((Collector_model.count / 500))
        staging_duration.append(0.02)
        staging_description.append(pred)
        print(pred, prob, "Time: %.3f ms" % (useTime * 1000), "%02d:%02d:%02d" % (h, m, s))
        # print("\r", pred, prob, "Time: %.3f seconds" % useTime, "%02d:%02d:%02d" % (h, m, s), end="")

        if pred == "N2" or pred == "N3":
            N2N3_Count += 1
        else:
            N2N3_Count = 0

        while Collector_model.count_SO <= Collector_model.count + int(fs * realtime_stride) and N2N3_Count >= 3:
            now_SO = time.time()
            data_SO = Collector_model.dataout_SO(so_stride)

            data_filt = SO_filter(b, a, data_SO, delay)

            data_tail_DownState = None
            data_tail_UpState = None

            data_tail = data_filt[-2500:]
            zero_index = []
            state = "Normal"

            # Find the negative zero points
            for i in range(len(data_tail) - 1):
                if data_tail[i] * data_tail[i + 1] <= 0 and data_tail[i] > 0:
                    zero_index.append(i)
                    zero_index.append(i + 1)
                    i += 1

            # The Last Zero Pointe
            try:
                zeroPoint = zero_index[-1]
            except BaseException:
                continue
            data_tail_DownState = data_tail[zeroPoint:]
            DownState_min = min(data_tail_DownState)

            # Detect the negative threshold
            if DownState_min <= neg_thresh:
                state = "DownState"
                index = np.where(data_tail_DownState == DownState_min)[0][0]
                for i in range(index, len(data_tail_DownState) - 1):
                    if i >= int(0.6 * fs):
                        break
                    # Find the positive zero point(only one)
                    if data_tail_DownState[i] * data_tail_DownState[i + 1] <= 0 and data_tail_DownState[i] <= 0:
                        data_tail_UpState = data_tail_DownState[i:]
                        UpState_max = data_tail_UpState[-1]

                        # Detect the positive threshold
                        if UpState_max >= pos_thresh:
                            if list(data_tail_UpState).index(UpState_max) > int(0.3 * fs):
                                break

                            state = "UpState"
                            so_onset.append(Collector_model.count_SO / fs)
                            so_duration.append(0.002)
                            so_description.append("SO_simu")
                            useTime_SO = float(time.time() - now_SO)
                            hms = timecal(Collector_model.count_SO / fs)

                            t = np.linspace(0, 30, 15000)
                            plt.plot(t, data_SO)
                            plt.plot(t[:], data_filt)
                            plt.legend(["Raw data", "Filtered data"])
                            # plt.plot(t[:], (data[4, Collector_model.count_SO - 15000: Collector_model.count_SO] - data[31, Collector_model.count_SO - 15000: Collector_model.count_SO]) * 1e6)
                            plt.hlines(y=pos_thresh, xmin=-5, xmax=35, colors="r")
                            plt.hlines(y=neg_thresh, xmin=-5, xmax=35, colors="r")
                            plt.hlines(y=0, xmin=-5, xmax=35, colors="r", linestyles="dashed")
                            plt.xlim([27, 30])
                            plt.xlabel("Time(s)")
                            plt.ylabel("Voltage(uV)")
                            plt.title("Up state, %s" % hms)
                            plt.savefig(SaveLabel + "/img/%s.png" % (hms))
                            plt.close()

                            Collector_model.count_SO += detectPauseInterval * fs
                            print(state, "Time: %.3f ms" % (useTime_SO * 1000))

except BaseException:
    pass

finally:
    realtime_model = np.array(realtime_model)
    usetime_model = np.array(usetime_model)
    annotations_staging = pd.DataFrame(
        {"onset": staging_onset, "duration": staging_duration, "description": staging_description})
    annotations_so = pd.DataFrame({"onset": so_onset, "duration": so_duration, "description": so_description})
    print()

    np.save(SaveLabel + "/Realtime_model.npy", realtime_model)
    np.save(SaveLabel + "/Realtime_model_time.npy", usetime_model)
    annotations_staging.to_csv(SaveLabel + "/Annotations_Staging.csv")
    annotations_so.to_csv(SaveLabel + "/Annotations_SO.csv")

print("Done! Use time: %s\n" % timecal(time.time() - now_realtime))
