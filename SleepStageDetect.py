import threading

import numpy as np
import torch
from SleepStageBlock.Model.Model import Model_total


class ThreadWithReturnValue(threading.Thread):
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        super().join()
        return self._return


class SleepStageModel:

    def __init__(self, device="cpu"):
        path = "SleepStageBlock/Model/Model.pth"
        self.model = Model_total()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self.model.to(device)
        self.device = device

        self.sleepDict = ["W/N1", "N2", "N3", "REM"]

    def predict(self, X):
        def func(X):
            X = torch.tensor(X, dtype=torch.float32).unsqueeze(dim=0).to(self.device)
            with torch.no_grad():
                output = self.model(X)[0]
                output_ind = torch.argmax(output).item()
                prob = {}
                for i in range(len(self.sleepDict)):
                    prob[self.sleepDict[i]] = "%.4f" % output[i].item()
            return self.sleepDict[output_ind], prob

        # f = threading.Thread(target=func, args=(X,))
        f = ThreadWithReturnValue(target=func, args=(X,))
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
