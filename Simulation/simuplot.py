from PyQt5.Qt import *
import pyqtgraph as pg
from pyqtgraph import PlotWidget
from PyQt5 import QtCore
import numpy as np
import pyqtgraph as pq

import mne

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


class Window(QWidget):
    def __init__(self):
        super().__init__()
        # 设置下尺寸
        self.resize(1200, 800)
        # 添加 PlotWidget 控件
        self.plotWidget = PlotWidget(self, background="w", title="C3-M2")
        # 设置该控件尺寸和相对位置
        self.plotWidget.setGeometry(QtCore.QRect(25, 25, 1150, 750))
        #
        pltItem = self.plotWidget.getPlotItem()
        #
        self.plotWidget.setYRange(-160, 100)
        left_axis = pltItem.getAxis("left")
        left_axis.setLabel("voltage", units="uV", **{"fontsize": "22pt"})
        bottom_axis = pltItem.getAxis("bottom")
        bottom_axis.setLabel("time", units="s", **{"fontsize": "22pt"})

        SaveLabel = "F:/35-2023_10_19-HongJiakang-Female"
        dataPath = SaveLabel + "/Sleep.vhdr"
        raw = mne.io.read_raw_brainvision(dataPath, preload=True)
        raw.pick(["C3", "M2"])
        raw.set_eeg_reference(["M2"])
        raw.pick(["C3"])
        raw.filter(0.1, 40)
        raw._data[0] = raw._data[0] * -1
        data, _ = raw[:]
        self.data = data[0] * 1e6
        self.count = 5000
        self.data_init = self.data[:self.count]

        self.x0 = 0
        self.xn = 10
        t = np.linspace(self.x0, self.xn, 5000)
        self.curve1 = self.plotWidget.ad
        self.curve1 = self.plotWidget.plot(t, self.data_init, pen="k")
        self.ptr1 = 0

        # 设定定时器
        self.timer = pq.QtCore.QTimer()
        # 定时器信号绑定 update_data 函数
        self.timer.timeout.connect(self.update_data)
        # 定时器间隔50ms，可以理解为 50ms 刷新一次数据
        self.timer.start(10)

    # 数据左移
    def update_data(self):
        # 数据填充到绘制曲线中
        self.data_init = np.concatenate([self.data_init[5:], self.data[self.count: self.count + 5]])
        self.count += 5

        t = np.linspace(self.x0, self.xn, 5000)
        self.curve1.setData(t, self.data_init)

        # x 轴记录点
        self.ptr1 += 0.01
        # 重新设定 x 相关的坐标原点
        self.curve1.setPos(self.ptr1, 0)


if __name__ == '__main__':
    import sys

    # PyQt5 程序固定写法
    app = QApplication(sys.argv)

    # 将绑定了绘图控件的窗口实例化并展示
    window = Window()
    window.show()

    # PyQt5 程序固定写法
    sys.exit(app.exec())
