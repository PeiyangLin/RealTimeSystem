import pandas as pd


class Subject:
    """
    Load the basic information of a single subject, which contain the name, id, gender and age.
    """

    def __init__(self):
        self.name = "Peiyang"
        self.id = 14
        self.gender = "Male"
        self.age = 22
        self.info = {}

    def __getDict__(self):
        self.info = {"Name": self.name,
                     "ID": "%03d" % self.id,
                     "Gender": self.gender,
                     "Age": "%d" % self.age}

    def show(self):
        self.__getDict__()
        for name in self.info:
            print("%s: %s" % (name, self.info[name]))

    def export(self):
        pass
