import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
import plotly.graph_objects as go
from libs.Models.ModelParent import ModelParent
import itertools


class DBSCANanomaly(ModelParent):
    def __init__(self, trainX: np.array, testX: np.array, testy: np.array, eps=0.15, min_samples=20):
        """
        :param trainX: Training data
        :param testX: Test data
        :param testy: Labels of the test data
        :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the
                    other
        :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point
        """
        super().__init__(trainX, testX, testy)
        self.eps = eps
        self.min_samples = min_samples
        self.concatX = np.concatenate([self.trainX, self.testX], axis=0)
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
        Initiates model and predicts based on the whole data set. Afterwards precision and recall of the model are
        calculated
        """
        self.standardize_dataset()
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)  # .fit_predict(self.concatX)
        self.testyPredicted = self.predict(self.concatX)
        self.precision = precision_score(self.testy, self.testyPredicted)
        self.recall = recall_score(self.testy, self.testyPredicted)
        self.specificity = recall_score(self.testy, self.testyPredicted, pos_label=0)

    def getROC(self):
        """
        Calculate specificity and recall for parameter combinations
        :return:
        Returns the mean distance between predicted and true anomalies as well as the data for the roc curve
        """
        # TODO Parameterräume wählen
        eps = ['gaussian', 'exponential']
        min_samples = [float(e) for e in np.arange(0.2, 1, 0.2)]
        parameters = [eps, min_samples]
        parameters = list(itertools.product(*parameters))
        roc = []
        distances = []
        for e in parameters:
            self.eps = e[0]
            self.min_samples = e[1]
            self.fit()
            roc.append({"parameter": e, "value": [float(self.recall), float(1 - self.specificity)]})
            distances.append({"parameter": e, "value": float(self.getStartDeltas())})
        return roc, distances

    def predict(self, values: np.ndarray) -> np.ndarray:
        """
        Divides the whole train-test set into clusters. Afterwards the "inlier cluster" is determined based on the
        assumption that it is the most abundant label in the outlier free training set. Every other cluster is
        labelled as an outlier cluster
        :param values: Values to determine the clusters on
        :return: Returns binary anomaly predictions of the test dataset
        """
        print(len(values))
        predictions = self.model.fit_predict(values)
        ## Efforts to make sure prediction output is binary ##
        trainlabels, traincounts = np.unique(predictions[:len(self.trainX)], return_counts=True)  # Lists all unique
        # labels and their count in trainX
        inlierlabel = trainlabels[np.argpartition(traincounts, -1)[-1:]]  #Returns most abundant Label in trainX, this
        # is the "inlier" label
        self.predictions = [1 if i != inlierlabel else 0 for i in predictions] #Iterates through all labels and marks
        # them as outliers unless they are equal to the inlierlabel
        return self.predictions[len(self.trainX):]

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

        title = "DBSCAN Recall: " + str(self.recall) + " Precision: " + str(
            self.precision) + " Epsilon: " + str(self.eps) + " Min_samples: " + str(self.min_samples) + "\n"
        fig.update_layout(title=title)
        fig.show()
