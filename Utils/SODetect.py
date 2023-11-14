import numpy as np
from scipy import signal
import threading


class ThreadWithReturnValue(threading.Thread):
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        super().join()
        return self._return


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

    factor = 1 * 1000 / fs

    Filter_Delay = round(Filter_Delay / factor)

    delay = np.zeros(Filter_Delay, dtype=np.float64)

    return b, a, N, Filter_Delay, delay


def SO_filter(b, a, x, Filter_Delay, delay):
    x = np.concatenate([x, delay])
    y = signal.lfilter(b, a, x)
    y = y[Filter_Delay:]
    return y


class SlowOscillationDetector:
    def __init__(self, wp, ws, fs, neg_thresh, pos_thresh, down_duration, up_duration, order=2):
        l_freq_p = wp[0]
        h_freq_p = wp[1]
        l_freq_s = ws[0]
        h_freq_s = ws[1]
        self.fs = fs
        self.b, self.a, self.N, self.Filter_Delay, self.delay = Filter_init(l_freq_p, h_freq_p, l_freq_s, h_freq_s, fs,
                                                                            order)

        self.neg_thresh = neg_thresh
        self.pos_thresh = pos_thresh

        self.down_duration = down_duration
        self.up_duration = up_duration

        self.UpState_N2 = 0
        self.UpState_N3 = 0

    def SO_find(self, data, sleepState):
        def func():
            data_filt = SO_filter(self.b, self.a, data, self.Filter_Delay, self.delay)
            data_tail = data_filt[-1500:]
            zero_index = []
            state = "Normal"

            # Find the negative zero points
            for i in range(len(data_tail) - 1):
                if data_tail[i] * data_tail[i + 1] <= 0 and data_tail[i] > 0:
                    zero_index.append(i)
                    zero_index.append(i + 1)
                    i += 1

            # The Last Zero Point
            try:
                zeroPoint = zero_index[-1]
            except:
                return state
            data_tail_DownState = data_tail[zeroPoint:]
            DownState_min = min(data_tail_DownState)

            # Detect the negative threshold
            if DownState_min <= self.neg_thresh:
                state = "DownState"
                index = np.where(data_tail_DownState == DownState_min)[0][0]
                for i in range(index, len(data_tail_DownState) - 1):
                    if i > int(self.down_duration * self.fs):
                        break
                    # Find the positive zero point(only one)
                    if data_tail_DownState[i] * data_tail_DownState[i + 1] <= 0 and data_tail_DownState[i] <= 0:
                        data_tail_UpState = data_tail_DownState[i:]
                        UpState_max = data_tail_UpState[-1]

                        # Detect the positive threshold
                        if UpState_max >= self.pos_thresh:
                            state = "UpState"
                            if sleepState == "N2":
                                self.UpState_N2 += 1
                            elif sleepState == "N3":
                                self.UpState_N3 += 1
                        break

            return state

        f = ThreadWithReturnValue(target=func, args=())
        f.start()
        SOstate = f.join()
        return SOstate

    def get_UpState(self, sleepState):
        if sleepState == "N2":
            return self.UpState_N2
        elif sleepState == "N3":
            return self.UpState_N3
