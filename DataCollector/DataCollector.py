import threading
import numpy as np
import pylsl
import time


class DataCollector:
    def __init__(self, C3=14, Eog=31, M2=18, fs=500, warmingTime=10):
        # Init the fixed parameters
        self.C3 = C3
        self.Eog = Eog
        self.M2 = M2
        self.fs = fs
        self.warmingTime = warmingTime * 60

        # first resolve an EEG stream on the lab network
        print("looking for an EEG stream...")
        streams = pylsl.resolve_stream('type', 'EEG')

        # create a new inlet to read from the stream
        self.inlet = pylsl.StreamInlet(streams[0])
        self.chunk_C3 = []
        self.chunk_REog = []
        self.chunk_M2 = []

        self.init_time = 0

        self.warming = False

    def dataCollect(self):
        def func():
            print("Data Collecting...")
            self.init_time = time.time()
            while True:
                sample, _ = self.inlet.pull_sample()

                self.chunk_C3.append(sample[self.C3] / 1e6)
                self.chunk_REog.append(sample[self.Eog] / 1e6)
                self.chunk_M2.append(sample[self.M2] / 1e6)

        threading.Thread(target=func, args=()).start()

    def warmDone(self):
        if time.time() - self.init_time > self.warmingTime:
            self.warming = True
        return None

    def dataGet(self, NeedTime):
        ind = int(-NeedTime * self.fs)
        data = [self.chunk_C3[ind:],
                self.chunk_REog[ind:],
                self.chunk_M2[ind:]]
        return np.array(data)

    def dataGet_All(self):
        length = min(len(self.chunk_C3), len(self.chunk_REog), len(self.chunk_M2))
        data = [self.chunk_C3[:length], self.chunk_REog[:length], self.M2[:length]]

        return np.array(data)
