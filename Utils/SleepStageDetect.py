import threading

import torch
from Utils.SleepStageBlock.Model.Model_One import Model_total as Model_One
from Utils.SleepStageBlock.Model.Model_Two import Model_total as Model_Two
from Utils.SleepStageBlock.Model.Model_Three import Model_total as Model_Three


class ThreadWithReturnValue(threading.Thread):
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        super().join()
        return self._return


class SleepStageModel:

    def __init__(self, useChannel=3, device="cpu"):
        if useChannel == 1:
            path = "Utils/SleepStageBlock/Model/One.pth"
            self.model = Model_One()
        elif useChannel == 2:
            path = "Utils/SleepStageBlock/Model/Two.pth"
            self.model = Model_Two()
        elif useChannel == 3:
            path = "Utils/SleepStageBlock/Model/Three.pth"
            self.model = Model_Three()
        self.model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        self.model.eval()

        self.device = device

        self.sleepDict = ["W/N1", "N2", "N3", "R"]

    def predict(self, x):
        def func():
            X = torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0).to(self.device)
            with torch.no_grad():
                output = self.model(X)[0]
                output_ind = torch.argmax(output).item()
                prob = {}
                for i in range(len(self.sleepDict)):
                    prob[self.sleepDict[i]] = "%.4f" % output[i].item()
            return self.sleepDict[output_ind], prob

        f = ThreadWithReturnValue(target=func, args=())
        f.start()
        pred, prob = f.join()
        return pred, prob

    def predict_offline(self, X):
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(dim=0).to(self.device)
        with torch.no_grad():
            output = self.model(X)[0]
            output_ind = torch.argmax(output).item()
            prob = {}
            for i in range(len(self.sleepDict)):
                prob[self.sleepDict[i]] = "%.4f" % output[i].item()
        return self.sleepDict[output_ind], prob


class SleepStageCollector:
    def __init__(self):
        self.N2_count = 0  # Init N2 Stage Count
        self.N3_count = 0  # Init N3 Stage Count
        self.R_count = 0  # Init REM Stage Count
        self.N2N3_count = 0  # Init N2/N3 Stage Count

    def update(self, stage):
        count_str = None
        if stage == "N2":
            self.N2_count += 1
            self.N2N3_count += 1
            count_str = "N2 Count: %d" % self.N2_count
        else:
            self.N2_count = 0

        if stage == "N3":
            self.N3_count += 1
            self.N2N3_count += 1
            count_str = "N3 Count: %d" % self.N3_count
        else:
            self.N3_count = 0

        if stage == "R":
            self.R_count += 1
            self.N2N3_count = 0
            count_str = "REM Count: %d" % self.R_count
        else:
            self.R_count = 0

        if stage == "W/N1":
            self.N2N3_count = 0

        return count_str
