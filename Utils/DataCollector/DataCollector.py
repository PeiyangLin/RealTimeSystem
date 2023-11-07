import threading
import numpy as np
import pylsl
import time
from scipy import signal


def Filter_init(l_freq_p, h_freq_p, l_freq_s, h_freq_s, fs, order):
    h_freq_p = h_freq_p * 2 / fs
    l_freq_p = l_freq_p * 2 / fs
    wp = [l_freq_p, h_freq_p]

    h_freq_s = h_freq_s * 2 / fs
    l_freq_s = l_freq_s * 2 / fs
    ws = [l_freq_s, h_freq_s]

    N, wn = signal.buttord(wp, ws, 5, 40)
    if order != 0:
        N = order

    b, a = signal.butter(N, wn, "bandpass")

    return b, a, N


def Stage_preProcess(b, a, x, factor):
    # DC Removal
    x = np.array(x)
    x = x * factor
    x = x - np.mean(x)

    # Filtered
    y = signal.filtfilt(b, a, x)
    return y


class DataCollector:
    def __init__(self, C3=14, Eog=31, EMG=32, M2=18, EMGREF=19, useChannel=3, amplifier="ANT", fs=500, warmingTime=30,
                 h_freq_p=40, l_freq_p=1, h_freq_s=50, l_freq_s=0.1, order=2):
        """

        :param C3: Channel of C3
        :param Eog: Channel of EOG
        :param EMG: Channel of EMG
        :param M2: Channel of M2
        :param fs: Sampling Rate
        :param warmingTime: Time Length of Buffer, unit is minutes
        """

        # Init the fixed parameters
        self.C3 = C3
        self.Eog = Eog
        self.M2 = M2
        self.EMG = EMG
        self.EMGREF = EMGREF
        self.useChannel = useChannel
        self.factor = 1 if amplifier == "BP" else 1e6
        self.fs = fs
        self.warmingTime = int(warmingTime * 60)
        self.bufferPoint = self.warmingTime * self.fs * (-1)
        self.bufferPoint_ = self.warmingTime * self.fs

        # first resolve an EEG stream on the lab network
        print("looking for an EEG stream...")
        streams = pylsl.resolve_stream('type', 'EEG')

        # create a new inlet to read from the stream
        self.inlet = pylsl.StreamInlet(streams[0])
        eeg_time_correction = self.inlet.time_correction()
        print("Time correction:", eeg_time_correction)

        self.chunk_C3 = []
        self.chunk_EOG = []
        self.chunk_EMG = []

        self.init_time = 0

        self.warming = False

        self.b, self.a, self.N = Filter_init(l_freq_p, h_freq_p, l_freq_s, h_freq_s, fs, order)

    def dataCollect(self):
        def func():
            print("Data Collecting...")
            self.init_time = time.time()
            while True:
                sample, _ = self.inlet.pull_sample()

                temp_M2 = sample[self.M2]
                temp_EMGREF = 0 if self.EMGREF == -1 else sample[self.EMGREF]

                temp_C3 = (sample[self.C3] - temp_M2)
                temp_Eog = (sample[self.Eog] - temp_M2)
                temp_EMG = (sample[self.EMG] - temp_EMGREF)

                self.chunk_C3.append(temp_C3)
                self.chunk_EOG.append(temp_Eog)
                self.chunk_EMG.append(temp_EMG)

        threading.Thread(target=func, args=()).start()

    def Buffer(self):
        if len(self.chunk_C3) <= self.bufferPoint_:
            return
        else:
            self.chunk_C3 = self.chunk_C3[self.bufferPoint:]
            self.chunk_EOG = self.chunk_EOG[self.bufferPoint:]
            self.chunk_EMG = self.chunk_EMG[self.bufferPoint:]

    def dataGet(self, NeedTime):
        self.Buffer()
        ind = int(-NeedTime * self.fs)
        data_preprocC3 = Stage_preProcess(self.b, self.a, self.chunk_C3, self.factor)
        data_preprocEOG = Stage_preProcess(self.b, self.a, self.chunk_EOG, self.factor)
        data_preprocEMG = Stage_preProcess(self.b, self.a, self.chunk_EMG, self.factor)

        data = None
        if self.useChannel == 1:
            data = [data_preprocC3[ind:]]
        elif self.useChannel == 2:
            data = [data_preprocC3[ind:],
                    data_preprocEOG[ind:]]
        elif self.useChannel == 3:
            data = [data_preprocC3[ind:],
                    data_preprocEOG[ind:],
                    data_preprocEMG[ind:]]
        return np.array(data)

    def dataGet_All(self):
        self.Buffer()
        length = min(len(self.chunk_C3), len(self.chunk_EOG), len(self.chunk_EMG))
        data = None
        if self.useChannel == 1:
            data = [self.chunk_C3[:length]]
        elif self.useChannel == 2:
            data = [self.chunk_C3[:length], self.chunk_EOG[:length]]
        elif self.useChannel == 3:
            data = [self.chunk_C3[:length], self.chunk_EOG[:length], self.chunk_EMG[:length]]
        return np.array(data)

    def dataGet_Single(self, NeedTime):
        self.Buffer()
        ind = int(-NeedTime * self.fs)
        data = self.chunk_C3[ind:]
        return np.array(data)


class DataCollector_temp:
    def __init__(self, EEG=[14], EOG=[31], EMG=[32], EEGREF=[18], EMGREF=19, useChannel=3, amplifier="ANT", fs=500,
                 warmingTime=30,
                 h_freq_p=40, l_freq_p=1, h_freq_s=50, l_freq_s=0.1, order=2):
        """

        :param EEG: Channel of C3
        :param EOG: Channel of EOG
        :param EMG: Channel of EMG
        :param EEGREF: Channel of eeg reference
        :param EMGREF: Channel of emg reference
        :param fs: Sampling Rate
        :param warmingTime: Time Length of Buffer, unit is minutes
        """

        # Init the fixed parameters
        self.EEG = EEG
        self.EOG = EOG
        self.M2 = EEGREF
        self.EMG = EMG
        self.EMGREF = EMGREF
        self.useChannel = useChannel
        self.factor = 1 if amplifier == "BP" else 1e6
        self.fs = fs
        self.warmingTime = int(warmingTime * 60)
        self.bufferPoint = self.warmingTime * self.fs * (-1)
        self.bufferPoint_ = self.warmingTime * self.fs

        # first resolve an EEG stream on the lab network
        print("looking for an EEG stream...")
        streams = pylsl.resolve_stream('type', 'EEG')

        # create a new inlet to read from the stream
        self.inlet = pylsl.StreamInlet(streams[0])
        eeg_time_correction = self.inlet.time_correction()
        print("Time correction:", eeg_time_correction)

        self.chunk_EEG = []
        for _ in self.EEG:
            self.chunk_EEG.append([])

        self.chunk_EOG = []
        for _ in self.chunk_EOG:
            self.chunk_EOG.append([])

        self.chunk_EMG = []

        self.init_time = 0

        self.warming = False

        self.b, self.a, self.N = Filter_init(l_freq_p, h_freq_p, l_freq_s, h_freq_s, fs, order)

    def dataCollect(self):
        def func():
            print("Data Collecting...")
            self.init_time = time.time()
            while True:
                sample, _ = self.inlet.pull_sample()

                temp_M2 = sample[self.M2]
                temp_EMGREF = 0 if self.EMGREF == -1 else sample[self.EMGREF]

                temp_C3 = (sample[self.C3] - temp_M2)
                temp_Eog = (sample[self.EOG] - temp_M2)
                temp_EMG = (sample[self.EMG] - temp_EMGREF)

                self.chunk_C3.append(temp_C3)
                self.chunk_EOG.append(temp_Eog)
                self.chunk_EMG.append(temp_EMG)

        threading.Thread(target=func, args=()).start()

    def Buffer(self):
        if len(self.chunk_C3) <= self.bufferPoint_:
            return
        else:
            self.chunk_C3 = self.chunk_C3[self.bufferPoint:]
            self.chunk_EOG = self.chunk_EOG[self.bufferPoint:]
            self.chunk_EMG = self.chunk_EMG[self.bufferPoint:]

    def dataGet(self, NeedTime):
        self.Buffer()
        ind = int(-NeedTime * self.fs)
        data_preprocC3 = Stage_preProcess(self.b, self.a, self.chunk_C3, self.factor)
        data_preprocEOG = Stage_preProcess(self.b, self.a, self.chunk_EOG, self.factor)
        data_preprocEMG = Stage_preProcess(self.b, self.a, self.chunk_EMG, self.factor)

        data = None
        if self.useChannel == 1:
            data = [data_preprocC3[ind:]]
        elif self.useChannel == 2:
            data = [data_preprocC3[ind:],
                    data_preprocEOG[ind:]]
        elif self.useChannel == 3:
            data = [data_preprocC3[ind:],
                    data_preprocEOG[ind:],
                    data_preprocEMG[ind:]]
        return np.array(data)

    def dataGet_All(self):
        self.Buffer()
        length = min(len(self.chunk_C3), len(self.chunk_EOG), len(self.chunk_EMG))
        data = None
        if self.useChannel == 1:
            data = [self.chunk_C3[:length]]
        elif self.useChannel == 2:
            data = [self.chunk_C3[:length], self.chunk_EOG[:length]]
        elif self.useChannel == 3:
            data = [self.chunk_C3[:length], self.chunk_EOG[:length], self.chunk_EMG[:length]]
        return np.array(data)

    def dataGet_Single(self, NeedTime):
        self.Buffer()
        ind = int(-NeedTime * self.fs)
        data = self.chunk_C3[ind:]
        return np.array(data)
