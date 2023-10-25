import PyQt5
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import mne
from scipy import signal
from SleepStageDetect import SleepStageModel as SleepModel

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


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
    x = np.concatenate([x, delay + x[-1]])
    y = signal.lfilter(b, a, x)
    y = y[Filter_Delay:]
    return y


def get_ticks(num):
    ms = num - int(num)
    useTime = int(num)
    m, s = divmod(useTime, 60)
    h, m = divmod(m, 60)
    s = s + ms
    return "%d:%d:%.3f" % (h, m, s)


b, a, N, Filter_Delay, delay = Filter_init(0.5, 4, 0.1, 5, 500, 2)

SaveLabel = "F:/35-2023_10_19-HongJiakang-Female"
dataPath = SaveLabel + "/Sleep.vhdr"

raw = mne.io.read_raw_brainvision(dataPath, preload=True)
raw.pick(["C3", "REOG", "M2"])
raw.set_eeg_reference(["M2"])
raw.pick(["C3", "REOG"])
raw.filter(0.1, 40)
raw._data[0] = raw._data[0] * -1
raw._data[1] = raw._data[1] * -1
data, _ = raw[:]

h = 0
m = 28
s = 40
data = data * 1e6
count = (3600 * h + 60 * m + s) * 500
duration = 15000
stride = 5
bias = 0.15
bias_so = 0.3
useChannel = 2
fs = 500
neg_thresh = -40
pos_thresh = 25
net = SleepModel(useChannel=useChannel)

win = pg.GraphicsLayoutWidget(show=True)
win.setWindowTitle('realtime data input')

data1 = data[0, count - duration: count]
data2 = SO_filter(b, a, data1, Filter_Delay, delay)

p1 = win.addPlot(row=1, col=1, title="C3-M2")
left_axis1 = p1.getAxis("left")
left_axis1.setLabel("voltage", units="uV", **{"font-size": "14pt"})
bottom_axis1 = p1.getAxis("bottom")
bottom_axis1.setLabel("time(s)", **{"font-size": "14pt"})
p1.setYRange(-100, 100)

p2 = win.addPlot(row=2, col=1, title="C3-M2, Filtered")
left_axis2 = p2.getAxis("left")
left_axis2.setLabel("voltage", units="uV", **{"font-size": "14pt"})
bottom_axis2 = p2.getAxis("bottom")
bottom_axis2.setLabel("time(s)", **{"font-size": "14pt"})
p2.setYRange(-100, 100)

x0 = 0
xn = 10
t = np.linspace(x0, xn, 5000)
ptr = 3600 * h + 60 * m + s - 10

data1 = data1[10000:]
data2 = data2[10000:]

curve1 = p1.plot(t, data1, pen="k")
curve1.setPos(ptr, 0)

curve2 = p2.plot(t, data1, pen="k")
curve2.setPos(ptr, 0)

neg_thresh_line = pg.PlotCurveItem(x=[ptr, ptr + 10], y=[neg_thresh, neg_thresh], pen=(255, 0, 0))
pos_thresh_line = pg.PlotCurveItem(x=[ptr, ptr + 10], y=[pos_thresh, pos_thresh], pen=(255, 0, 0))
zero_thresh_line = pg.PlotCurveItem(x=[ptr, ptr + 10], y=[0, 0], pen=(255, 0, 0))
p2.addItem(neg_thresh_line)
p2.addItem(pos_thresh_line)
p2.addItem(zero_thresh_line)

line_x = []
lines1 = []
lines2 = []

font = PyQt5.QtGui.QFont("Arial", 12)

N2N3_Count = 0
SO_restrict = 0


def update1():
    global data1, data2, ptr, count, N2N3_Count, SO_restrict, neg_thresh_line, pos_thresh_line
    if len(line_x) > 0:
        if line_x[0] - bias_so < ptr:
            p1.removeItem(lines1[0][0])
            p1.removeItem(lines1[0][1])
            p2.removeItem(lines2[0][0])
            p2.removeItem(lines2[0][1])
            del lines1[0]
            del lines2[0]
            del line_x[0]

    data1 = data[0, count - duration + stride: count + stride]
    data2 = SO_filter(b, a, data1, Filter_Delay, delay)

    if count % 1500 == 0:
        pred, _ = net.predict_offline(data[:, count - duration + stride: count + stride])
        if pred == "N2" or pred == "N3":
            N2N3_Count += 1
        else:
            N2N3_Count = 0
        temp = ptr + 10
        line1 = pg.PlotCurveItem(x=[temp, temp], y=[-200, 200], pen=(255, 0, 255))
        label1 = pg.TextItem(text=pred, color=(255, 0, 255))
        label1.setFont(font)
        label1.setPos(temp - bias, -95)

        line2 = pg.PlotCurveItem(x=[temp, temp], y=[-200, 200], pen=(255, 0, 255))
        label2 = pg.TextItem(text=pred, color=(255, 0, 255))
        label2.setFont(font)
        label2.setPos(temp - bias, -95)

        line_x.append(temp)
        lines1.append([line1, label1])
        lines2.append([line2, label2])
        p1.addItem(lines1[-1][0])
        p1.addItem(lines1[-1][1])
        p2.addItem(lines2[-1][0])
        p2.addItem(lines2[-1][1])

    if N2N3_Count >= 3 and count >= SO_restrict:
        data_filt = data2

        data_tail = data_filt[-2500:]
        zero_index = []

        # Find the negative zero points
        for i in range(len(data_tail) - 1):
            if data_tail[i] * data_tail[i + 1] <= 0 and data_tail[i] > 0:
                zero_index.append(i)
                zero_index.append(i + 1)
                i += 1

        # The Last Zero Pointe
        try:
            zeroPoint = zero_index[-1]
        except BaseException:
            zeroPoint = -1
        data_tail_DownState = data_tail[zeroPoint:]
        DownState_min = min(data_tail_DownState)

        # Detect the negative threshold
        if DownState_min <= neg_thresh:
            index = np.where(data_tail_DownState == DownState_min)[0][0]
            for i in range(index, len(data_tail_DownState) - 1):
                if i >= int(0.6 * fs):
                    break
                # Find the positive zero point(only one)
                if data_tail_DownState[i] * data_tail_DownState[i + 1] <= 0 and data_tail_DownState[i] <= 0:
                    data_tail_UpState = data_tail_DownState[i:]
                    UpState_max = data_tail_UpState[-1]

                    # Detect the positive threshold
                    if UpState_max >= pos_thresh:
                        if list(data_tail_UpState).index(UpState_max) > int(0.3 * fs):
                            break

                        state = "SO_U"
                        SO_restrict = count + fs

                        temp = ptr + 10
                        line1 = pg.PlotCurveItem(x=[temp, temp], y=[-200, 200], pen=(0, 0, 255))
                        label1 = pg.TextItem(text=state, color=(0, 0, 255))
                        label1.setFont(font)
                        label1.setPos(temp - bias_so, -95)

                        line2 = pg.PlotCurveItem(x=[temp, temp], y=[-200, 200], pen=(0, 0, 255))
                        label2 = pg.TextItem(text=state, color=(0, 0, 255))
                        label2.setFont(font)
                        label2.setPos(temp - bias_so, -95)

                        line_x.append(temp)
                        lines1.append([line1, label1])
                        lines2.append([line2, label2])
                        p1.addItem(lines1[-1][0])
                        p1.addItem(lines1[-1][1])
                        p2.addItem(lines2[-1][0])
                        p2.addItem(lines2[-1][1])

    ptr += 0.01

    count += stride
    neg_thresh_line.setData([ptr, ptr + 10], [neg_thresh, neg_thresh])
    pos_thresh_line.setData([ptr, ptr + 10], [pos_thresh, pos_thresh])
    zero_thresh_line.setData([ptr, ptr + 10], [0, 0])

    data1 = data1[10000:]
    data2 = data2[10000:]

    curve1.setData(t, data1)
    curve1.setPos(ptr, 0)

    curve2.setData(t, data2)
    curve2.setPos(ptr, 0)


timer = pg.QtCore.QTimer()
timer.timeout.connect(update1)
timer.start(10)

if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
