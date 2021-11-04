import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score
import plotly.graph_objects as go
from libs.Models.ModelParent import ModelParent
import itertools

class LOFAnomaly(ModelParent):
    def __init__(self, trainX: np.array, testX: np.array, testy: np.array, n_neighbors=20, contamination="auto", amplifier = 3):
        """
        :param trainX: Training data
        :param testX: Test data
        :param testy: Labels of the test data
        :param n_neighbors: Number of neighbors to be used
        :param contamination: Contamination in the dataset. Either "auto" or in range [0,0.5]
        """
        super().__init__(trainX, testX, testy)
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = None
        self.precision = 0
        self.recall = 0
        self.predictions = []
        self.amplifier = amplifier

    def standardize_dataset(self):
        """
        Standardizes dataset (mean=0, std=1) according to training data
        """
        scaler = StandardScaler().fit(self.trainX)
        self.trainX = scaler.transform(self.trainX)
        self.testX = scaler.transform(self.testX)

    def fit(self) -> None:
        """
        Fits model with training data. Calulcates precision and recall of the model
        """
        self.standardize_dataset()
        self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=self.contamination, novelty=True).fit(self.trainX)
        self.testyPredicted = self.predict(self.testX)[0]
        self.precision = precision_score(self.testy, self.testyPredicted)
        self.recall = recall_score(self.testy, self.testyPredicted)
        self.specificity = recall_score(self.testy, self.testyPredicted, pos_label=0)

    def predict(self, values: np.ndarray) -> np.ndarray:
        """
        Calculates anomaly scores of the test data based on the training data
        :param values: Values to determine anomaly score on
        :return: Returns binary anomaly predictions and raw anomaly scores
        """
        scores = self.model.decision_function(values)
        #self.predictions = self.model.predict(values)
        #self.predictions = [1 if e == -1 else 0 for e in self.predictions]
        #threshold = np.abs(scores.mean()) + 3*scores.std()
        threshold = np.abs(scores[:500].mean()) + self.amplifier * scores[:500].std()
        self.predictions = [1 if np.abs(e) > threshold else 0 for e in scores]
        return self.predictions, scores

    def getROC(self):
        """
        Calculate specificity and recall for parameter combinations
        :return:
        Returns the mean distance between predicted and true anomalies as well as the data for the roc curve
        """
        #TODO Parameterräume wählen
        n_neighbors = [5, 10, 20, 30, 50, 80]
        contamination = [0.001, 0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.21, 0.34, 0.5]
        amplifierlist = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        parameters = [n_neighbors, contamination, amplifierlist]
        parameters = list(itertools.product(*parameters))
        roc = []
        distances = []
        for e in parameters:
            self.n_neighbors = e[0]
            self.contamination = e[1]
            self.amplifier = e[2]
            self.fit()
            roc.append({"parameter": e,"value": [float(self.recall), float(1- self.specificity)]})
            distances.append({"parameter": e, "value": float(self.getStartDeltas())})
        return roc, distances

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
        Plots the LOF Decision Function
        Also shows performance of the model by displaying performance metrics as well as a plot of the training,
        test and prediction distribution.
        """
        fig = go.Figure()
        x0 = self.trainX.reshape(1, -1)[0]
        x1 = self.testX.reshape(1, -1)[0]
        fig.add_trace(go.Scatter(x=x0, y=[0.5 for e in range(len(self.trainX))],
                                 name="Training data", mode="markers", marker_color="blue"))
        fig.add_trace(go.Scatter(x=x1, y=self.testy,
                                 name="test data true", mode="markers", marker_color="red"))
        fig.add_trace(go.Scatter(x=x1, y=self.predictions,
                                 name="test data predicted", mode="markers", marker_color="yellow"))

        min = np.min(self.trainX)
        max = np.max(self.trainX)
        xpredict = np.array(np.arange(min, max, 0.1)).reshape(-1,1)
        y = self.predict(xpredict)[1]
        x = np.array(np.arange(min, max, 0.1))
        fig.add_trace(go.Scatter(x = x, y = y, marker_color="orange", name="Local Outlier Factor"))
        title = "LOF Recall: " + str(self.recall) + " Precision: " + str(self.precision) + "\n" + "n_neighbors: " \
                + str(self.n_neighbors) + " Contamination: " + str(self.contamination)
        fig.update_layout(title=title)
        fig.show()

