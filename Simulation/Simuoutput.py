import numpy as np
import mne


class dataSim:

    def __init__(self, dataPath, C3=2, REog=6, EMG=7, TimeSet=3600):
        self.data = np.load(dataPath)
        self.C3 = C3
        self.REog = REog
        self.EMG = EMG
        self.fs = 500
        self.count = 0
        self.warm = self.fs * TimeSet

    def dataout(self, need_time, sleepTime):
        self.count += self.fs * sleepTime
        if self.count >= len(self.data[0]):
            raise RuntimeError("error")
        data_now = self.data[:, self.count - self.warm: self.count]
        if len(data_now[0]) == 0:
            data_now = self.data[:, :self.count]

        data_C3 = data_now[self.C3]
        data_REOG = data_now[self.REog]
        data_EMG = data_now[self.EMG]

        data_output = np.array([data_C3, data_REOG, data_EMG])

        data_output = data_output[:, -(need_time * self.fs):]

        nowTime = self.count / self.fs
        m, s = divmod(nowTime, 60)
        h, m = divmod(m, 60)

        return data_output, (h, m, s)

    def warmup(self, warmTime):
        self.count += self.fs * warmTime


class dataSim_yasa:

    def __init__(self, dataPath, C3=2, REog=6, EMG=7, TimeSet=3600):
        self.data = np.load(dataPath)
        self.C3 = C3
        self.REog = REog
        self.EMG = EMG
        self.fs = 500
        self.count = 0
        self.warm = self.fs * TimeSet

    def dataout(self, sleepTime):
        self.count += self.fs * sleepTime
        if self.count >= len(self.data[0]):
            raise RuntimeError('testError')
        data_now = self.data[:, self.count - self.warm: self.count]
        if len(data_now[0]) == 0:
            data_now = self.data[:, :self.count]

        data_C3 = data_now[self.C3]
        data_REOG = data_now[self.REog]
        data_output = np.array([data_C3 / 1e6, data_REOG / 1e6])

        info = mne.create_info(ch_names=['C3', 'REOG'],
                               ch_types='eeg',
                               sfreq=self.fs)

        raw = mne.io.RawArray(data_output, info, verbose=False)

        nowTime = self.count / self.fs
        m, s = divmod(nowTime, 60)
        h, m = divmod(m, 60)

        return raw, (h, m, s)

    def warmup(self, warmTime):
        self.count += self.fs * warmTime
