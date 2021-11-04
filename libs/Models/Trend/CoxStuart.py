from builtins import NotImplementedError

import pandas as pd
import numpy as np
import math

from scipy.stats import norm
from libs.Models.ModelParent import ModelParent
from lib.Utils.Functions import FemtoScore as FemtoScore

class CoxStuart(ModelParent):
    def __init__(self, trainX: np.array, testX: np.array, testy: np.array, window = 50, alpha = 0.0001, debug = False):
        super().__init__(trainX, testX, testy)
        self.window = window    #Window was used as n in predict, was replaced by length of the time series
        self.alpha = alpha
        self.debug = debug
        self.recall = 0
        self.precision = 0
        self.model = None #Save the object which will be trained/ used for predictions in here

    def fit(self) -> None:
        # Für die ganze oder für steps?
        # Wie lang ist ein Step?
        stepsize = 7*24*6
        TP,FP,TN,FN = 0, 0, 0, 0
        self.testX = self.testX.reshape(1,-1)[0]
        preds = []
        femtoscores = []
        for i in range(48, len(self.testX), stepsize):
            testData = self.testX[0:i]
            testY = set(self.testy[0:i])
            if 1 in testY:
                testY = 1
            else:
                testY = 0
            prediction, pvalue = self.predict(testData)
            preds.append(prediction)
            if prediction == testY:
                if testY == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if testY == 1:
                    FN += 1
                else:
                    FP += 1
            if prediction == 1:
                #preds geht nur alle 48 steps, testy je einen step
                femtoscores.append(FemtoScore(self.testy[0:i:stepsize],preds))
        self.precision = TP / (TP + FP)
        self.recall = TP / (TP + FN)
        self.femtoscore = float(np.mean(femtoscores))



    def predict(self, values: np.ndarray) -> np.ndarray:
        """
        Cox-Stuart criterion
        H0: trend exists
        H1: otherwise
        Detects linear trends
        Optimal window = 50
        """
        n = len(values)
        idx = np.arange(1, n + 1)
        X = pd.Series(values, index = idx)

        S1 = [(n - 2 * i) if X[i] <= X[n - i + 1] else 0 for i in range(1, n // 2)]
        n = float(n)
        S1_ = (sum(S1) - n ** 2 / 8) / math.sqrt(n * (n ** 2 - 1) / 24)
        u = norm.ppf(1 - self.alpha / 2)
        if self.debug:
            print('|S1*|:', abs(S1_))
            print("u:", u)

        return abs(S1_) > u, abs(S1_)  # H0 accept

    def showResults(self):
        """
        Put code to explain the model here
        """

        raise NotImplementedError
