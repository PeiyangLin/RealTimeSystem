# Author: LangaLinn
# File Create: 2023/10/23 17:53

import numpy as np
import pandas as pd
from scipy import signal
import mne
import yasa
import matplotlib.pyplot as plt


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


def SO_filter(b, a, x):
    y = signal.filtfilt(b, a, x)
    return y


def total_analysis(dataSet, fs=500):
    down_min_set = []
    up_max_set = []
    down_duration_set = []
    up_duration_set = []
    for i in range(len(dataSet)):
        data = dataSet[i]
        down_min = min(data[:fs])
        up_max = max(data[fs:])
        down_duration = 0.
        up_duration = 0.

        for j in range(fs - 1):
            if data[j] * data[j + 1] <= 0 <= data[j]:
                down_duration = (fs - j) / fs
                break

        for j in range(fs - 1):
            if data[fs + j] * data[fs + j + 1] <= 0 and data[fs + j] >= 0:
                up_duration = j / fs
                break

        down_min_set.append(down_min)
        up_max_set.append(up_max)
        down_duration_set.append(down_duration)
        up_duration_set.append(up_duration)


def avg_analysis(data, fs=500):
    down_min = min(data[:fs])
    up_max = max(data[fs:])
    down_duration = 0.
    up_duration = 0.

    for i in range(fs - 1):
        if data[i] * data[i + 1] <= 0 <= data[i]:
            down_duration = (fs - i) / fs
            break

    for i in range(fs - 1):
        if data[fs + i] * data[fs + i + 1] <= 0 and data[fs + i] >= 0:
            up_duration = i / fs
            break

    print("Minimum of down state: %.3fuV" % down_min)
    print("Maximum of up state: %.3fuV" % up_max)
    print("Duration of down state: %.1fms" % (down_duration * 1000))
    print("Duration of up state: %.1fms" % (up_duration * 1000))
    print()

    return down_min, up_max, down_duration, up_duration


Amplifier = "BP"
pick_channel = "C3"
ref_channel = "M2"
fs = 500
b, a, N, Filter_Delay, delay = Filter_init(0.5, 4, 0.1, 5, fs, 2)

SaveLabel = "D:/sleep/080"
dataPath = SaveLabel + "/TMR.vhdr"
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

raw = mne.io.read_raw_brainvision(dataPath, preload=True)
raw.pick([pick_channel, ref_channel])
raw.set_eeg_reference([ref_channel])
raw.filter(0.1, 40)
raw.pick([pick_channel])

# sw = yasa.sw_detect(data=raw, sf=500, ch_names=["C3"], freq_sw=[0.5, 4], amp_neg=[40, 200], amp_pos=[10, 150])
sw = yasa.sw_detect(data=raw, sf=500, ch_names=["C3"])
slow_Wave_offline = sw.summary()

onset = np.array(slow_Wave_offline["Start"]) * fs
onset = onset.astype(int)

duration = np.array(slow_Wave_offline["Duration"]) * fs
duration = duration.astype(int)

offline_description = []
for i in range(len(slow_Wave_offline["Start"])):
    offline_description.append("SO_yasa")

data, _ = raw[:]
data = data[0] * 1e6
data_filt = SO_filter(b, a, data)

so_wave_set = []
so_wave_filt_set = []

for i in range(len(onset)):
    startPoint = onset[i]
    for j in range(duration[i] - 1):
        point = startPoint + j
        if data_filt[point] * data_filt[point + 1] <= 0 and data_filt[point] <= 0:
            so_wave = data[point - 500: point + 500]
            so_wave_filt = data_filt[point - 500: point + 500]

            so_wave_set.append(so_wave)
            so_wave_filt_set.append(so_wave_filt)
            break
so_wave_set = np.array(so_wave_set)
so_wave_filt_set = np.array(so_wave_filt_set)
print(so_wave_set.shape)

so_wave_avg = np.mean(so_wave_set, axis=0)
so_wave_filt_avg = np.mean(so_wave_filt_set, axis=0)
avg_analysis(so_wave_avg)
avg_analysis(so_wave_filt_avg)

t = np.linspace(-1, 1, 1000)
plt.plot(t, so_wave_avg)
plt.plot(t, so_wave_filt_avg)
plt.hlines(0, xmin=-2, xmax=2, colors="r")
plt.xlim([-1, 1])
plt.legend(["data_raw", "data_filt"])
plt.xlabel("time(s)")
plt.ylabel("voltage(uV)")
plt.title("Mean of slow wave")
plt.show()
