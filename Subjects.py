import pandas as pd
from Utils.Prepare.SlowWave_prepare import prepare
import os


class Subject:
    """
    Load the basic information of a single subject, which contain the name, id, gender and age.
    """

    def __init__(self):
        self.name = "Test"
        self.id = 88
        self.gender = "Female"
        self.age = 20
        self.info = {}
        self.date = "2023_10_26"

        self.__getDict__()
        self.logDir = "LOG/%02d-%s-%s-%s" % (self.id, self.date, self.name, self.gender)

    def __getDict__(self):
        self.info = {"Name": self.name,
                     "ID": "%03d" % self.id,
                     "Gender": self.gender,
                     "Age": "%d" % self.age,
                     "Date": "%s" % self.date}

    def get_fileName(self):
        return self.logDir

    def show(self):
        self.__getDict__()
        for name in self.info:
            print("%s: %s" % (name, self.info[name]))

    def export(self):
        pass

    def subjectInit(self, loadLabel=None):
        isExists = os.path.exists(self.logDir)

        if not isExists:
            os.mkdir(self.logDir)
            down_min_avg, up_max_avg, down_duration_avg, up_duration_avg = prepare(loadLabel, self.logDir)
            self.info["Down_min_avg"] = down_min_avg
            self.info["Up_max_avg"] = up_max_avg
            self.info["Down_duration_avg"] = down_duration_avg
            self.info["Up_duration_avg"] = up_duration_avg
            for key in self.info:
                print(self.info[key])
            info = pd.DataFrame(self.toDataFrame())
            info.to_csv(self.logDir + "/SubjectInfo.csv")
        else:
            info = pd.read_csv(self.logDir + "/SubjectInfo.csv")
            down_min_avg = info["Down_min_avg"][0]
            up_max_avg = info["Up_max_avg"][0]
            down_duration_avg = info["Down_duration_avg"][0]
            up_duration_avg = info["Up_duration_avg"][0]
        return down_min_avg, up_max_avg, down_duration_avg, up_duration_avg

    def toDataFrame(self):
        frame = self.info.copy()
        for key in frame:
            frame[key] = [frame[key]]
        return frame
