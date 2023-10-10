import threading
import numpy as np
import pylsl
import time
from scipy import signal


class Filter_PARA:
    def __init__(self):
        self.h_freq_p = 40
        self.l_freq_p = 1

        self.h_freq_s = 50
        self.l_freq_s = 0.1

        self.fs = 500

        self.order = 2

    def paraInput(self):
        return self.h_freq_p, self.l_freq_p, self.h_freq_s, self.l_freq_s, self.fs, self.order


filtPara = Filter_PARA()


def Filter_init(l_freq_p, h_freq_p, l_freq_s, h_freq_s, fs, order):
    h_freq_p = h_freq_p * 2 / fs
    l_freq_p = l_freq_p * 2 / fs
    wp = [l_freq_p, h_freq_p]

    h_freq_s = h_freq_s * 2 / fs
    l_freq_s = l_freq_s * 2 / fs
    ws = [l_freq_s, h_freq_s]

    N, wn = signal.buttord(wp, ws, 5, 40)
    if N != 0:
        N = order

    b, a = signal.butter(N, wn, "bandpass")

    return b, a, N


def Stage_filter(b, a, x):
    y = signal.filtfilt(b, a, x)
    return y


class DataCollector_EEG:
    def __init__(self, C3=14, M2=18, fs=500, warmingTime=30):
        """

        :param C3: Channel of C3
        :param M2: Channel of M2
        :param fs: Sampling Rate
        :param warmingTime: Time Length of Buffer, unit is minutes
        """

        # Init the fixed parameters
        self.C3 = C3
        self.M2 = M2
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

        self.init_time = 0

        self.warming = False

        l_freq_p, h_freq_p, l_freq_s, h_freq_s, fs, order = filtPara.paraInput()
        self.b, self.a, self.N = Filter_init(l_freq_p, h_freq_p, l_freq_s, h_freq_s, fs, order)

    def dataCollect(self):
        def func():
            print("Data Collecting...")
            self.init_time = time.time()
            while True:
                sample, _ = self.inlet.pull_sample()

                temp_M2 = sample[self.M2]

                temp_C3 = (sample[self.C3] - temp_M2)

                self.chunk_C3.append(temp_C3)

        threading.Thread(target=func, args=()).start()

    def Buffer(self):
        if len(self.chunk_C3) <= self.bufferPoint_:
            return
        else:
            self.chunk_C3 = self.chunk_C3[self.bufferPoint:]

    def dataGet(self, NeedTime):
        self.Buffer()
        ind = int(-NeedTime * self.fs)
        data_filterC3 = Stage_filter(self.b, self.a, self.chunk_C3)
        data = [data_filterC3[ind:]]
        return np.array(data)

    def dataGet_All(self):
        self.Buffer()
        length = len(self.chunk_C3)
        data = [self.chunk_C3[:length]]
        return np.array(data)

    def dataGet_Single(self, NeedTime):
        self.Buffer()
        ind = int(-NeedTime * self.fs)
        data = self.chunk_C3[ind:]
        return np.array(data)


class DataCollector_EOG:
    def __init__(self, C3=14, Eog=31, M2=18, fs=500, warmingTime=30):
        """

        :param C3: Channel of C3
        :param Eog: Channel of EOG
        :param M2: Channel of M2
        :param fs: Sampling Rate
        :param warmingTime: Time Length of Buffer, unit is minutes
        """

        # Init the fixed parameters
        self.C3 = C3
        self.Eog = Eog
        self.M2 = M2
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
        self.chunk_Eog = []

        self.init_time = 0

        self.warming = False

        l_freq_p, h_freq_p, l_freq_s, h_freq_s, fs, order = filtPara.paraInput()
        self.b, self.a, self.N = Filter_init(l_freq_p, h_freq_p, l_freq_s, h_freq_s, fs, order)

    def dataCollect(self):
        def func():
            print("Data Collecting...")
            self.init_time = time.time()
            while True:
                sample, _ = self.inlet.pull_sample()
                # print(len(sample))

                temp_M2 = sample[self.M2]

                temp_C3 = (sample[self.C3] - temp_M2)
                temp_Eog = (sample[self.Eog] - temp_M2)

                self.chunk_C3.append(temp_C3)
                self.chunk_Eog.append(temp_Eog)

        threading.Thread(target=func, args=()).start()

    def Buffer(self):
        if len(self.chunk_C3) <= self.bufferPoint_:
            return
        else:
            self.chunk_C3 = self.chunk_C3[self.bufferPoint:]
            self.chunk_Eog = self.chunk_Eog[self.bufferPoint:]

    def dataGet(self, NeedTime):
        self.Buffer()
        ind = int(-NeedTime * self.fs)
        data_filterC3 = Stage_filter(self.b, self.a, self.chunk_C3)
        data_filterEOG = Stage_filter(self.b, self.a, self.chunk_Eog)
        data = [data_filterC3[ind:],
                data_filterEOG[ind:]]

        return np.array(data)

    def dataGet_All(self):
        self.Buffer()
        length = min(len(self.chunk_C3), len(self.chunk_Eog))
        data = [self.chunk_C3[:length], self.chunk_Eog[:length]]
        return np.array(data)

    def dataGet_Single(self, NeedTime):
        self.Buffer()
        ind = int(-NeedTime * self.fs)
        data = self.chunk_C3[ind:]
        return np.array(data)


class DataCollector_ALL:
    def __init__(self, C3=14, Eog=31, EMG=32, M2=18, fs=500, warmingTime=30):
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
        self.chunk_Eog = []
        self.chunk_EMG = []

        self.init_time = 0

        self.warming = False

        l_freq_p, h_freq_p, l_freq_s, h_freq_s, fs, order = filtPara.paraInput()
        self.b, self.a, self.N = Filter_init(l_freq_p, h_freq_p, l_freq_s, h_freq_s, fs, order)

    def dataCollect(self):
        def func():
            print("Data Collecting...")
            self.init_time = time.time()
            while True:
                sample, _ = self.inlet.pull_sample()

                temp_M2 = sample[self.M2]

                temp_C3 = (sample[self.C3] - temp_M2)
                temp_Eog = (sample[self.Eog] - temp_M2)
                temp_EMG = (sample[self.EMG])

                self.chunk_C3.append(temp_C3)
                self.chunk_Eog.append(temp_Eog)
                self.chunk_EMG.append(temp_EMG)

        threading.Thread(target=func, args=()).start()

    def Buffer(self):
        if len(self.chunk_C3) <= self.bufferPoint_:
            return
        else:
            self.chunk_C3 = self.chunk_C3[self.bufferPoint:]
            self.chunk_Eog = self.chunk_Eog[self.bufferPoint:]
            self.chunk_EMG = self.chunk_EMG[self.bufferPoint:]

    def dataGet(self, NeedTime):
        self.Buffer()
        ind = int(-NeedTime * self.fs)
        data_filterC3 = Stage_filter(self.b, self.a, self.chunk_C3)
        data_filterEOG = Stage_filter(self.b, self.a, self.chunk_Eog)
        data_filterEMG = Stage_filter(self.b, self.a, self.chunk_EMG)
        data = [data_filterC3[ind:],
                data_filterEOG[ind:],
                data_filterEMG[ind:]]
        return np.array(data)

    def dataGet_All(self):
        self.Buffer()
        length = min(len(self.chunk_C3), len(self.chunk_Eog), len(self.chunk_EMG))
        data = [self.chunk_C3[:length], self.chunk_Eog[:length], self.chunk_EMG[:length]]
        return np.array(data)

    def dataGet_Single(self, NeedTime):
        self.Buffer()
        ind = int(-NeedTime * self.fs)
        data = self.chunk_C3[ind:]
        return np.array(data)
