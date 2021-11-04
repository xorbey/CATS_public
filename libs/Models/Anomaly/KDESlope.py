import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from libs.Models.ModelParent import ModelParent  #Put your ModelParent path here
import plotly.graph_objects as go
import itertools

class KDESlope(ModelParent):
    """
    Kernel Density Estimator Klasse, mit flexibler Einstellung der Grenzwahrscheinlichkeit, des kernels und der Bandbreite
    """
    def __init__(self, trainX: np.array, testX: np.array, testy: np.array, probability = 0.01, kernel = "gaussian", bandwidth = 0.5):
        """

        :param trainX: Trainingsdaten - Features
        :param testX: Testdatenfeatures
        :param testy: Labels der Testdaten
        :param probability: Grenzwahrscheinlichkeit, ab der ein Sample als Ausreißer eingeordnet werden soll
        :param kernel: Kernel zur Bestimmung des KDEs
        :param bandwidth: Bandbreite zur Bestimmung des KDEs
        """
        super().__init__(trainX, testX, testy)
        self.probability = probability
        self.model = None #Save the object which will be trained/ used for predictions in here
        self.precision = 0
        self.recall = 0
        self.kernel = kernel
        self.bandwidth = bandwidth

    def fit(self) -> None:
        """
        Fits Trainingsdata zu KDE, berechnet Recall und Precision auf Basis der Testdaten
        """
        if self.trainX.shape[1] != 2:
            diffs = np.diff(self.trainX.reshape(1, -1)[0], prepend=self.trainX[0]).reshape(-1, 1)
            self.trainX = np.append(self.trainX, diffs, axis=1)
        if self.testX.shape[1] != 2:
            diffs = np.diff(self.testX.reshape(1, -1)[0], prepend=self.testX[0]).reshape(-1, 1)
            self.testX = np.append(self.testX, diffs, axis=1)
        self.model = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(self.trainX)
        self.testyPredicted, scores = self.predict(self.testX)
        self.precision = precision_score(self.testy, self.testyPredicted)
        self.recall = recall_score(self.testy, self.testyPredicted)
        self.specificity = recall_score(self.testy, self.testyPredicted, pos_label = 0)

    def getROC(self):
        """
        Calculate specificity and recall for parameter combinations
        :return:
        Returns the mean distance between predicted and true anomalies as well as the data for the roc curve
        """
        kernels = ['gaussian', 'exponential']
        bandwidth = [float(e) for e in np.arange(0.2,1, 0.2)]
        conf = [float(e) for e  in np.arange(0.01, 0.1, 0.02)]
        parameters = [kernels, bandwidth, conf]
        parameters = list(itertools.product(*parameters))
        roc = []
        distances = []
        for e in parameters:
            self.kernel = e[0]
            self.bandwidth = e[1]
            self.probability = e[2]
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


        # get Parameter Range
        # calc with Dataset
        # store precision / recall values



    def predict(self, values: np.ndarray) -> np.ndarray:
        """

        :param values: Werte für die eine Vorhersage getroffen werden soll, entweder reine Zeitserie oder bereits mit Ableitungen
        :return: gibt Klassifikationsergebnisse zurück
        """
        #assuming input is a time series
        if values.ndim == 2:
            scores = np.exp(self.model.score_samples(values))
            self.predictions = [0 if (score > self.probability) else 1 for score in scores ]
            return self.predictions, scores
        if values.ndim == 1:
            diff = np.diff(values, prepend = values[0]).reshape(-1,1)
            values = np.append(values.reshape(-1,1), diff, axis = 1)
            scores = np.exp(self.model.score_samples(values))
            self.predictions = [0 if (score > self.probability) else 1 for score in scores]
            return self.predictions, scores
        else:
            print("Wrong input dimensions: " + str(values.shape))
            return

    def showResults(self):
        """
        Erstellt einen Plot der den KDE darstellt, die Trainings und Testdatenverteilung und die vorhergesagten labels
        Funktioniert nur für den 1D-Fall
        :return:
        """
        fig = go.Figure()
        x0 = [e[0] for e in self.trainX]
        y0 = [e[1] for e in self.trainX]

        minx = np.min(x0)
        maxx = np.max(x0)
        miny = np.min(y0)
        maxy = np.max(y0)
        stepx = (maxx - minx)/100
        stepy = (maxy - miny)/100
        xpredict = np.mgrid[minx:maxx:stepx, miny:maxy:stepy].reshape(2, -1).T[:10000]
        dummy ,y = self.predict(xpredict)
        predictions = y.reshape(100,100)

        fig.add_trace(go.Contour(x = np.arange(minx, maxx, stepx)[:100], y =np.arange(miny, maxy, stepy)[:100], z = predictions))
        fig.add_trace(go.Scatter(x=x0, y=y0,
                                 name="Training data", mode="markers", marker_color="blue"))
        title = "KDE Recall: " + str(self.recall) + " Precision: " + str(self.precision)\
                + " Probability: " + str(self.probability) + " Kernel: " + str(self.kernel) + " Bandwidth: " \
                + str(self.bandwidth)
        fig.update_layout(title = title, xaxis_title="Value", yaxis_title="Slope")
        fig.show()
        colordic = {0: "green", 1: "red"}
        fig2 = go.Figure(go.Scatter(x=[e for e in range(len(self.testX))], y=self.testX[:,0], mode="markers",
                                   marker_color=[colordic[e] for e in self.predict(self.testX)[0]]))
        fig2.show()