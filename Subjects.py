import pandas as pd


class Subject:
    """
    Load the basic information of a single subject, which contain the name, id, gender and age.
    """

    def __init__(self):
        self.name = "ZhouJianyang"
        self.id = 1
        self.gender = "Male"
        self.age = 26
        self.info = {}
        self.date = "2023_10_24"

        self.__getDict__()

    def __getDict__(self):
        self.info = {"Name": self.name,
                     "ID": "%03d" % self.id,
                     "Gender": self.gender,
                     "Age": "%d" % self.age,
                     "Date": "%s" % self.date}

    def get_fileName(self):
        fileName = "%02d-%s-%s-%s" % (self.id, self.date, self.name, self.gender)
        return fileName

    def show(self):
        self.__getDict__()
        for name in self.info:
            print("%s: %s" % (name, self.info[name]))

    def export(self):
        pass
