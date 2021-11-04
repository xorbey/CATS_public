import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import precision_score, recall_score
import plotly.graph_objects as go
from libs.Models.ModelParent import ModelParent
import itertools

class ARMA(ModelParent):
    def __init__(self, trainX: np.array, testX: np.array, testy: np.array, ARlags=5, MAlags=10, threshold=3):
        """
        :param trainX: Trainingset
        :param testX: Testset
        :param testy: Labels of the testset
        :param ARlags: Number of autoregression lags to be used. Can be integer or list of integers
        :param MAlags: Number of moving average lags to be used. Can be integer or list of integers
        :param threshold: Anomaly threshold in Range [0,1] #TODO Threshold mit 0,1 Threshold kompatibel machen
        """
        super().__init__(trainX, testX, testy)
        self.ARlags = ARlags
        self.MAlags = MAlags
        self.threshold = threshold
        self.model = None #Save the object which will be trained/ used for predictions in here
        self.precision = 0
        self.recall = 0
        self.predictions = []

    def standardize_dataset(self):
        """
        Standardizes dataset (mean=0, std=1) according to training dataset
        """
        scaler = StandardScaler().fit(self.trainX)
        self.trainX = scaler.transform(self.trainX)
        self.testX = scaler.transform(self.testX)

    def fit(self) -> None:
        """
        Fits model with training dataset. Calculates precision and recall of the model.
        """
        self.standardize_dataset()
        self.model = ARIMA(self.trainX, order=(self.ARlags, 0, self.MAlags)).fit()
        self.testyPredicted = self.predict(self.testX)[0]
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
        ARlags = ['gaussian', 'exponential']
        MAlags = [float(e) for e in np.arange(0.2,1, 0.2)]
        threshold = [float(e) for e  in np.arange(0.01, 0.1, 0.02)]
        parameters = [ARlags, MAlags, threshold]
        parameters = list(itertools.product(*parameters))
        roc = []
        distances = []
        for e in parameters:
            self.ARlags = e[0]
            self.MAlags = e[1]
            self.threshold = e[2]
            self.fit()
            roc.append({"parameter": e,"value": [float(self.recall), float(1- self.specificity)]})
            distances.append({"parameter": e, "value": float(self.getStartDeltas())})
        return roc, distances

    def predict(self, values: np.ndarray) -> np.ndarray:
        """
        Forecasts training dataset and calculates errors based on param values.
        :param values: Values used to evaluate the error of the predictions
        :return: Returns anomaly predictions in binary format and raw anomaly scores.
        """
        predicts = self.model.predict(start=len(self.trainX), end=len(self.trainX)+len(values)-1)
        scores = - np.abs(np.asarray(values.reshape((len(values),)) - predicts)) + self.threshold
        self.predictions = [1 if (score < 0) else 0 for score in scores]
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
        Plots the ARMA Decision Function
        Also shows performance of the model by displaying performance metrics as well as a plot of the training,
        test and prediction distribution.
        """
        fig = go.Figure()
        x0 = self.trainX.reshape(1, -1)[0]
        x1 = self.testX.reshape(1, -1)[0]
        fig.add_trace(go.Scatter(x=x0, y=[0.5 for e in range(len(self.trainX))],
                                 name="Training data", mode="markers", marker_color="blue"))
        fig.add_trace(go.Scatter(x=x1, y=self.testy,
                                 name="Test Data true Labels", mode="markers", marker_color="red"))
        fig.add_trace(go.Scatter(x=x1, y=self.predictions,
                                 name="Test Data predicted Labels", mode="markers", marker_color="yellow"))

        min = np.min(self.trainX)
        max = np.max(self.trainX)
        xpredict = np.array(np.arange(min, max, 0.1)).reshape(-1, 1)
        y = self.predict(xpredict)[1]
        x = np.array(np.arange(min, max, 0.1))
        fig.add_trace(go.Scatter(x=x, y=y, marker_color="orange", name="ARMA decision function"))
        title = "ARMA Recall: " + str(self.recall) + " Precision: " + str(
            self.precision) + " ARLags: " + str(self.ARlags) + " MALags: " + str(self.MAlags) + " Threshold: " + \
            str(self.threshold) + "\n"
        fig.update_layout(title=title)
        fig.show()
        colors = {1: "red", 0: "green"}
        x = [e for e in range(len(self.testX))]
        y = [e[0] for e in self.testX]
        fig = go.Figure(go.Scatter(x=x, y=y, mode="markers",
                                   marker_color=[colors[e] for e in self.testyPredicted]))
        fig.show()
