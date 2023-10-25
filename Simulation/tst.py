import pyqtgraph as pg
import numpy as np
import psutil
import datetime


# 获取CPU使用率的定时回调函数
def get_cpu_info():
    global time_flag1, time_flag1_1, time_flag1_2, time_flag1_3
    try:
        time_flag1_3 = time_flag1_2
    except:
        time_flag1_3 = 0
    try:
        time_flag1_2 = time_flag1_1
    except:
        time_flag1_2 = 0
    try:
        time_flag1_1 = time_flag1
        print(time_flag1)
    except:
        time_flag1_1 = 0

    # xax = p1.getAxis('left') # 改成坐标轴y
    time_flag1 = datetime.datetime.now().strftime('%H:%M:%S.%f')  # 含微秒的日期时间，来源 比特量化

    xax = p.getAxis('bottom')  # 坐标轴x
    ticks = [list(zip(range(4), (time_flag1_3, time_flag1_2, time_flag1_1, time_flag1)))]  # 声明五个坐标，分别是
    xax.setTicks(ticks)

    if len(data_list) < historyLength:
        data_list.append(float(5))
    else:
        data_list[:-1] = data_list[1:]  # 前移
        data_list[-1] = float(5)

    plot.setData(data_list, pen='g')


if __name__ == '__main__':
    data_list = []

    # pyqtgragh初始化
    # 创建窗口
    app = pg.mkQApp()  # 建立app
    win = pg.GraphicsWindow()  # 建立窗口
    win.setWindowTitle(u'pyqtgraph 实时波形显示工具')
    win.resize(800, 500)  # 小窗口大小

    # 创建图表
    historyLength = 4  # 横坐标长度
    p = win.addPlot()  # 把图p加入到窗口中
    p.showGrid(x=True, y=True)  # 把X和Y的表格打开
    # p.setRange(xRange=[0, historyLength], yRange=[0, 100], padding=0)

    p.setLabel(axis='left', text='CPU利用率')  # 靠左
    p.setLabel(axis='bottom', text='时间')
    p.setTitle('CPU利用率实时数据')  # 表格的名字
    plot = p.plot()

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(get_cpu_info)  # 定时刷新数据显示
    timer.start(100)  # 多少ms调用一次

    app.exec_()
