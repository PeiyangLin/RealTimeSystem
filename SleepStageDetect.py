import threading

import torch
from SleepStageBlock.Model.Model_One import Model_total as Model_One
from SleepStageBlock.Model.Model_Two import Model_total as Model_Two
from SleepStageBlock.Model.Model_Three import Model_total as Model_Three


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
            path = "SleepStageBlock/Model/One.pth"
            self.model = Model_One()
        elif useChannel == 2:
            path = "SleepStageBlock/Model/Two.pth"
            self.model = Model_Two()
        elif useChannel == 3:
            path = "SleepStageBlock/Model/Three.pth"
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