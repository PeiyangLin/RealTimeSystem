import threading
import numpy as np
import pylsl
import time


class DataCollector:
    def __init__(self, C3=14, Eog=31, EMG=32, M2=18, fs=500, warmingTime=30):
        # Init the fixed parameters
        self.C3 = C3
        self.Eog = Eog
        self.M2 = M2
        self.EMG = EMG
        self.fs = fs
        self.warmingTime = int(warmingTime * 60)

        # first resolve an EEG stream on the lab network
        print("looking for an EEG stream...")
        streams = pylsl.resolve_stream('type', 'EEG')

        # create a new inlet to read from the stream
        self.inlet = pylsl.StreamInlet(streams[0])
        self.chunk_C3 = []
        self.chunk_Eog = []
        self.chunk_EMG = []

        # 30 Min sti
        # self.chunk_C3 = list(np.zeros(60 * fs * 60))
        # self.chunk_Eog = list(np.zeros(60 * fs * 60))
        # self.chunk_M2 = list(np.zeros(60 * fs * 60))

        self.init_time = 0

        self.warming = False

    def dataCollect(self):
        def func():
            print("Data Collecting...")
            self.init_time = time.time()
            while True:
                sample, _ = self.inlet.pull_sample()

                temp_M2 = sample[self.M2]
                temp_C3 = (sample[self.C3] - temp_M2) / 1e6
                temp_Eog = (sample[self.Eog] - temp_M2) / 1e6
                temp_EMG = (sample[self.EMG]) / 1e6

                self.chunk_C3.append(temp_C3)
                self.chunk_Eog.append(temp_Eog)
                self.chunk_EMG.append(temp_EMG)

        threading.Thread(target=func, args=()).start()

    def Buffer(self):
        if len(self.chunk_C3) <= (self.warmingTime * self.fs):
            return
        else:
            self.chunk_C3 = self.chunk_C3[-self.warmingTime * self.fs:]
            self.chunk_Eog = self.chunk_Eog[-self.warmingTime * self.fs:]
            self.chunk_EMG = self.chunk_EMG[-self.warmingTime * self.fs:]


    def dataGet(self, NeedTime):
        self.Buffer()
        ind = int(-NeedTime * self.fs)
        data = [self.chunk_C3[ind:],
                self.chunk_Eog[ind:],
                self.chunk_EMG[ind:]]
        print(len(self.chunk_C3))
        return np.array(data)

    def dataGet_All(self):
        self.Buffer()
        length = min(len(self.chunk_C3), len(self.chunk_Eog), len(self.chunk_EMG))
        data = [self.chunk_C3[:length], self.chunk_Eog[:length], self.chunk_EMG[:length]]
        return np.array(data)
