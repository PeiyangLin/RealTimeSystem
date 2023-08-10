import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Data
path = "../Data/SS2/01-02-0001 PSG.edf"
mask = "../Data/SS2/01-02-0001 Base.edf"
spindle = "../Data/SS2/01-02-0001 Spindles_E2.edf"

# Raw Data Input
raw = mne.io.read_raw_edf(path, preload=True)
mask = mne.read_annotations(mask)
spindle = mne.read_annotations(spindle)
raw.set_annotations(mask)
raw.set_annotations(spindle)
# print(raw.info["sfreq"])
# print(raw.info["ch_names"])
# print(spindle)
pick = ["EEG C3-CLE"]
raw.pick(pick)
scalings = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=5e-4,
                emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
                resp=1, chpi=1e-4, whitened=1e2)


# Data Slice
data, _ = raw[:]
start = 12455
# start = 9550
duration = 1200
fs = 256
data_slice = data[0, (start * fs):((start + duration) * fs)]
# raw.plot(start=start, duration=60, scalings=scalings)

# Data Filtered
Wn_low = 2 * 0.16 / fs
Wn_high = 2 * 1.28 / fs

b, a = signal.butter(2, [Wn_low, Wn_high], 'bandpass')
filtedData = signal.filtfilt(b, a, data_slice)

# New Data Plotting

# new_data = [data_slice, filtedData]
# new_info = mne.create_info(["Raw", "Filtered"], fs, ch_types='eeg')
# new_raw = mne.io.RawArray(new_data, new_info)
# new_raw.plot(duration=30, scalings=scalings)
# plt.show()

# Zero Line Define
length = len(filtedData)
zeroLine = np.zeros(length)
time = (np.array(range(length)) / fs) + start

# Zero Points
zero_time = []
zero_dat = []
for i in range(1, len(filtedData)):
    now = filtedData[i]
    prev = filtedData[i - 1]
    if (now >= 0 > prev) or (now <= 0 < prev):
        zero_time.append(time[i - 1])
        zero_time.append(time[i])
        zero_dat.append(prev)
        zero_dat.append(now)



# Data Plotting
# plt.plot(time, filtedData, ".")
plt.plot(time, filtedData)
plt.plot(time, zeroLine)
plt.plot(zero_time, zero_dat, "r.")
plt.plot(time, data_slice, linewidth=0.8)

# Spindle Waves
SpindleOnset = spindle.onset
SpindleDuration = spindle.duration
spindleTime = []
spindleData = []
for i in range(len(SpindleOnset)):
    if start <= SpindleOnset[i] <= start+duration:
        tempStart = SpindleOnset[i]
        tempDuration = SpindleDuration[i]
        startPoint = int(tempStart * fs)
        durationLength = int(tempDuration * fs)

        spindleTime = list(np.array(range(startPoint, startPoint+durationLength)) / fs)
        spindleData = list(data[0, startPoint:startPoint+durationLength])
        plt.plot(spindleTime, spindleData, "m", linewidth=1.2)

plt.xlim([start, start + 10])
plt.ylim([-100e-6, 100e-6])

plt.xlabel("Time(s)")
plt.ylabel("Voltage(uV)")

plt.title("EEG Slice %ds to %ds" % (start, start+duration))

plt.legend(["Filtered Data", "Zero Line", "Zero Points", "Raw Data", "Spindle Waves"])

plt.show()
