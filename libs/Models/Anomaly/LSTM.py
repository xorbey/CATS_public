import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
from scipy.stats import norm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from libs.Models.ModelParent import ModelParent
import itertools


class LSTMAnomaly(ModelParent):
    def __init__(self, trainX: np.array, testX: np.array, testy: np.array, n_epochs=20, threshold=0.999, split=5):
        """
        :param trainX: Training data
        :param testX: Test data
        :param testy: Labels of the test data
        :param n_epochs: Training epochs
        :param threshold: Anomaly threshold in range [0,1]
        :param split: How many predecessors the model considers for one forecast
        """
        super().__init__(trainX, testX, testy)
        self.n_epochs = n_epochs
        self.threshold = threshold
        self.split = split
        self.concatX = np.concatenate([trainX, testX], axis=0)
        self.model = None  # Save the object which will be trained/ used for predictions in here
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
        """
        self.scaler = StandardScaler().fit(self.trainX)
        self.trainX = self.scaler.transform(self.trainX)
        self.testX = self.scaler.transform(self.testX)
        """


        Xmin = self.trainX.min()
        Xmax = self.trainX.max()
        self.trainX = (self.trainX-Xmin)/(Xmax-Xmin)
        self.testX = (self.testX-Xmin)/(Xmax-Xmin)



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
        train_size = int(len(self.trainX))
        XX, Xy = self.create_Xy_dataset(np.concatenate([self.trainX, self.testX], axis=0), self.split)
        #XX, Xy = self.create_Xy_dataset(self.concatX, self.split)
        trainXX, testXX = XX[:train_size - self.split], XX[train_size - self.split:]
        trainXy, testXy = Xy[:train_size - self.split], Xy[train_size - self.split:]
        input_shape = (self.split, 1)
        trainXX = trainXX.reshape((trainXX.shape[0], trainXX.shape[1], 1))
        testXX = testXX.reshape((testXX.shape[0], testXX.shape[1], 1))

        ## Building the network
        self.model = Sequential()
        self.model.add(LSTM(4, input_shape=input_shape, return_sequences=True))
        self.model.add(LSTM(4))
        self.model.add(Dense(1))
        self.model.compile(optimizer="adam", loss="mse")

        self.model.fit(trainXX, trainXy, epochs=self.n_epochs, shuffle=False, verbose = 0)

        self.testyPredicted = self.predict(testfeatures=testXX, truevalues=testXy)[0]

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
        n_epochs = [10, 20, 50]
        threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.95, 0.98, 0.99, 0.999]
        parameters = [n_epochs, threshold]
        parameters = list(itertools.product(*parameters))
        roc = []
        distances = []
        for e in parameters:
            self.n_epochs = e[0]
            self.threshold = e[1]
            self.fit()
            roc.append({"parameter": e,"value": [float(self.recall), float(1- self.specificity)]})
            distances.append({"parameter": e, "value": float(self.getStartDeltas())})
        return roc, distances

    def predict(self, testfeatures: np.ndarray, truevalues: np.ndarray) -> np.ndarray:
        """
        Forecasts the dataset based on observed values. Calulates errors based on the truevalues
        :param testfeatures: Lookback dataset of the test values
        :param truevalues: true test values
        :return:
        """
        predicts = self.model.predict(testfeatures)
        scores = - self.AnomalyScore(np.square(truevalues - predicts)) + self.threshold #.reshape((len(truevalues),))
        self.predictions = [1 if (score < 0) else 0 for score in scores]
        plt.plot(np.arange(len(truevalues)), truevalues, label="TrueValues")
        plt.plot(np.arange(len(predicts)), predicts, label="PredictedValues")
        plt.plot(np.arange(len(scores)), scores, label="Anomalyscores")
        plt.plot(np.arange(len(self.predictions)), self.predictions, label="Predictions")
        plt.legend()
        plt.show()
        return self.predictions, scores

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

