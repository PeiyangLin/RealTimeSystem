import threading
import numpy as np
import pylsl
import time


class DataCollector:
    def __init__(self, C3=14, Eog=31, M2=18, fs=500, warmingTime=30):
        # Init the fixed parameters
        self.C3 = C3
        self.Eog = Eog
        self.M2 = M2
        self.fs = fs
        self.warmingTime = int(warmingTime * 60)

        # first resolve an EEG stream on the lab network
        print("looking for an EEG stream...")
        streams = pylsl.resolve_stream('type', 'EEG')

        # create a new inlet to read from the stream
        self.inlet = pylsl.StreamInlet(streams[0])
        # self.chunk_C3 = []
        # self.chunk_Eog = []
        # self.chunk_M2 = []

        # 30 Min sti
        self.chunk_C3 = list(np.zeros(30 * fs * 60))
        self.chunk_Eog = list(np.zeros(30 * fs * 60))
        self.chunk_M2 = list(np.zeros(30 * fs * 60))

        self.init_time = 0

        self.warming = False

    def dataCollect(self):
        def func():
            print("Data Collecting...")
            self.init_time = time.time()
            while True:
                sample, _ = self.inlet.pull_sample()

                self.chunk_C3.append(sample[self.C3] / 1e6)
                self.chunk_Eog.append(sample[self.Eog] / 1e6)
                self.chunk_M2.append(sample[self.M2] / 1e6)

                self.Buffer()

        threading.Thread(target=func, args=()).start()

    def Buffer(self):
        if len(self.chunk_C3) <= (self.warmingTime * self.fs):
            return
        else:
            self.warming = True
            del self.chunk_C3[0]
            del self.chunk_Eog[0]
            del self.chunk_M2[0]

    def dataGet(self, NeedTime):
        ind = int(-NeedTime * self.fs)
        data = [self.chunk_C3[ind:],
                self.chunk_Eog[ind:],
                self.chunk_M2[ind:]]
        return np.array(data)

    def dataGet_All(self):
        # now = time.time()

        length = min(len(self.chunk_C3), len(self.chunk_Eog), len(self.chunk_M2))
        data = [self.chunk_C3[:length], self.chunk_Eog[:length], self.chunk_M2[:length]]
        # print("Collect Time: %.3fms" % (float(time.time()-now) * 1000))

        return np.array(data)
