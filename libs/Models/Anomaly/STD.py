import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
from scipy.stats import norm
import plotly.graph_objects as go

from libs.Models.ModelParent import ModelParent
import itertools


class STD(ModelParent):
    def __init__(self, trainX: np.array, testX: np.array, testy: np.array, threshold=0.999, split=5):
        """
        :param trainX: Training data
        :param testX: Test data
        :param testy: Labels of the test data
        :param n_epochs: Training epochs
        :param threshold: Anomaly threshold in range [0,1]
        :param split: How many predecessors the model considers for one forecast
        """
        super().__init__(trainX, testX, testy)
        self.threshold = threshold
        self.split = split

        self.precision = 0
        self.recall = 0
        self.predictions = []
        self.specificity = 0

    def create_Xy_dataset(self, sequence, steps):
        """
        Splits the whole dataset (train+test) into a 2D array X of shape (len(sequence),steps) and a 1D array y of
        len(sequence). X consists of lookback values for each elements of y.
        :param sequence: univariate dataset
        :param steps: Number of lookback values for each element of y
        :return: X,y
        """
        X, y = [], []
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + steps
            # check if we are beyond the sequence
            if end_ix > len(sequence) - 1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.asarray(X), np.asarray(y)

    def standardize_dataset(self):
        """
        Standardizes dataset (mean=0, std=1) according to training data
        """
        self.scaler = StandardScaler().fit(self.trainX)
        self.trainX = self.scaler.transform(self.trainX)
        self.testX = self.scaler.transform(self.testX)

    def AnomalyScore(self, rawscore):
        std = np.std(self.trainX)   # = 1
        mean = np.mean(self.trainX) # = 0
        zscore = abs((rawscore - mean) / std)
        anomalyscore = (norm.cdf(zscore)-norm.cdf(-zscore))
        return anomalyscore

    def fit(self) -> None:
        """
        Initializes the LSTM. The goal of the NN is to forecast one value of the time series based on the last
        observations.
        It's fitted using trainXX (the observed "lookback" values) and trainXy (the "labels").
        Finally, precision and recall of the model are calculated
        """
        self.standardize_dataset()

        self.std = np.std(self.trainX)
        self.mean = np.mean(self.trainX)


        self.testyPredicted = self.predict(self.testX)

        self.precision = precision_score(self.testy, self.testyPredicted)
        self.recall = recall_score(self.testy, self.testyPredicted)
        self.specificity = recall_score(self.testy, self.testyPredicted, pos_label=0)

    def getROC(self):
        """
        Calculate specificity and recall for parameter combinations
        :return:
        Returns the mean distance between predicted and true anomalies as well as the data for the roc curve
        """
        #TODO Parameterräume wählen

        threshold = np.arange(1,3.2,0.2)
        parameters = threshold
        roc = []
        distances = []
        for e in parameters:
            self.threshold = e
            self.fit()
            roc.append({"parameter": e,"value": [float(self.recall), float(1- self.specificity)]})
            distances.append({"parameter": e, "value": float(self.getStartDeltas())})
        return roc, distances

    def predict(self, testfeatures: np.ndarray) -> np.ndarray:
        """
        Forecasts the dataset based on observed values. Calulates errors based on the truevalues
        :param testfeatures: Lookback dataset of the test values
        :param truevalues: true test values
        :return:
        """
        threshold = self.std*self.threshold
        results = [1 if np.abs(e) > np.abs(self.mean) + threshold else 0 for e in testfeatures]
        return results

    def getStartDeltas(self):
        """
        Überprüfe für jede Anomalie nach wie vielen Schritten eine Anomalie erkannt
        wurde, falls diese erkannt wurde, miss die Distanz
        :return:
        gibt den Mittelwert der Distanzen zurück
        """
        result = []
        for e in enumerate(self.testy):
            if e[1] == 1:
                for el in list(enumerate(self.testyPredicted))[e[0]:]:
                    if el[1] == 1:
                        result.append(el[0] - e[0])
                        break
        return np.mean(result)

    def showResults(self):
        """
        Plots the performance of the model by displaying performance metrics as well as a test and prediction
        distribution
        """
        fig = go.Figure()
        x0 = self.trainX.reshape(1, -1)[0]
        x1 = self.testX.reshape(1, -1)[0]
        fig.add_trace(go.Scatter(x=x0, y=[0.5 for e in range(len(self.trainX))],
                                 name="Training data", mode="markers", marker_color="blue"))
        fig.add_trace(go.Scatter(x=x1, y=self.testy,
                                 name="test labels True", mode="markers", marker_color="red"))
        fig.add_trace(go.Scatter(x=x1, y=self.predictions,
                                 name="test labels predicted", mode="markers", marker_color="yellow"))
        title = "LSTM Recall: " + str(self.recall) + " Precision: " + str(
            self.precision) + " n_epochs: " + str(self.n_epochs) + " threshold: " \
            + str(self.threshold) + " split: " + str(self.split) + "\n"
        fig.update_layout(title=title)
        fig.show()

