import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import precision_score, recall_score
import plotly.graph_objects as go
from libs.Models.ModelParent import ModelParent
import itertools

class AR(ModelParent):
    def __init__(self, trainX: np.array, testX: np.array, testy: np.array, ARlags=10, threshold=3):
        """
        :param trainX: Training data
        :param testX: Test data
        :param testy: Labels of the test data
        :param ARlags: Number of lags to be used. Can be integer or list
        :param threshold: Anomaly threshold in Range [0,1] #TODO Threshold mit 0,1 Threshold kompatibel machen
        """
        super().__init__(trainX, testX, testy)
        self.ARlags = ARlags
        self.threshold = threshold
        self.model = None
        self.precision = 0
        self.recall = 0
        self.predictions = []

    def standardize_dataset(self):
        """
        Standardizes dataset (mean=0, std=1) according to training data
        """
        scaler = StandardScaler().fit(self.trainX)
        self.trainX = scaler.transform(self.trainX)
        self.testX = scaler.transform(self.testX)

    def fit(self) -> None:
        """
        Fits model with training data. Calculates precision and recall of the model.
        """
        self.standardize_dataset()
        self.model = AutoReg(self.trainX, lags=self.ARlags, old_names=False).fit()
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
        threshold = [float(e) for e in np.arange(0.01, 0.1, 0.02)]
        parameters = [ARlags, threshold]
        parameters = list(itertools.product(*parameters))
        roc = []
        distances = []
        for e in parameters:
            self.ARlags = e[0]
            self.threshold = e[1]
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
        predicts = self.model.predict(start=len(self.trainX), end=len(self.trainX) + len(values) - 1)
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
        Plots the AutoRegression Decision Function
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
        fig.add_trace(go.Scatter(x=x, y=y, marker_color="orange", name="Autoregression decision function"))
        title = "Autoregression Recall: " + str(self.recall) + " Precision: " + str(
            self.precision) + " ARLags: " + str(self.ARlags) + " Threshold: " + str(self.threshold) + "\n"
        fig.update_layout(title=title)
        fig.show()
