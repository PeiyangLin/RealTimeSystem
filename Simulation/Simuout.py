import numpy as np


class dataSim:

    def __init__(self, dataPath=None, data=None, C3=2, REog=6, M2=7, TimeSet=300, SO_window=30):
        if dataPath is not None:
            self.data = np.load(dataPath)
        else:
            self.data = data

        self.data = self.data * 1e6

        self.C3 = C3
        self.REog = REog
        self.M2 = M2
        self.fs = 500
        self.count = 0
        self.count_SO = 0
        self.warm = self.fs * TimeSet
        self.SO_Window = SO_window * self.fs

        self.data[C3] = self.data[C3] - self.data[M2]

    def dataout(self, need_time, sleepTime):
        self.count += self.fs * sleepTime
        self.count_SO = self.count
        if self.count >= len(self.data[0]):
            raise RuntimeError("error")
        data_now = self.data[:, self.count - self.warm: self.count]
        if len(data_now[0]) == 0:
            data_now = self.data[:, :self.count]

        data_C3 = data_now[self.C3]
        data_REOG = data_now[self.REog]

        data_output = np.array([data_C3, data_REOG])

        data_output = data_output[:, -(need_time * self.fs):]

        nowTime = self.count / self.fs
        m, s = divmod(nowTime, 60)
        h, m = divmod(m, 60)

        return data_output, (h, m, s)

    def dataout_SO(self, sleepTime):
        if self.count >= len(self.data[0]):
            raise RuntimeError("error")
        data_now = self.data[:, self.count_SO - self.SO_Window: self.count_SO]
        if len(data_now[0]) == 0:
            data_now = self.data[:, :self.count]

        data_C3 = data_now[self.C3]
        data_output = np.array(data_C3)

        nowTime = self.count / self.fs

        m, s = divmod(nowTime, 60)
        h, m = divmod(m, 60)
        self.count_SO += int(self.fs * sleepTime / 1000)

        return data_output

    def warmup(self, warmTime):
        self.count += self.fs * warmTime
