import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from libs.Models.ModelParent import ModelParent  #Put your ModelParent path here
import plotly.graph_objects as go
import itertools
from sklearn.preprocessing import StandardScaler


class KDE(ModelParent):
    """
    Kernel Density Estimator Klasse, mit flexibler Einstellung der Grenzwahrscheinlichkeit, des kernels und der Bandbreite
    """
    def __init__(self, trainX: np.array, testX: np.array, testy: np.array, probability = 0.01, kernel = "gaussian", bandwidth = 0.5, split = 5):
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
        self.split = split

    def standardize_dataset(self):
        """
        Standardizes dataset (mean=0, std=1) according to training data
        """
        scaler = StandardScaler().fit(self.trainX)
        self.trainX = scaler.transform(self.trainX)
        self.testX = scaler.transform(self.testX)

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

    def fit(self) -> None:
        """
        Fits Trainingsdata zu KDE, berechnet Recall und Precision auf Basis der Testdaten
        """
        self.standardize_dataset()
        train_size = int(len(self.trainX))
        XX, Xy = self.create_Xy_dataset(np.concatenate([self.trainX, self.testX], axis=0), self.split)
        trainXX, testXX = XX[:train_size - self.split], XX[train_size - self.split:]
        trainXy, testXy = Xy[:train_size - self.split], Xy[train_size - self.split:]
        input_shape = (self.split, 1)
        trainXX = trainXX.reshape((trainXX.shape[0], trainXX.shape[1]))
        testXX = testXX.reshape((testXX.shape[0], testXX.shape[1]))

        self.model = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(trainXX)
        self.testyPredicted, scores = self.predict(testXX)
        self.precision = precision_score(self.testy, self.testyPredicted)
        self.recall = recall_score(self.testy, self.testyPredicted)
        self.specificity = recall_score(self.testy, self.testyPredicted, pos_label = 0)

    def getRoc(self):
        kernels = ['gaussian', 'exponential', 'epanechnikov']
        bandwidth = [0.2, 0.3, 0.5, 0.8, 1.3, 2.1, 3.4, 5.5]
        threshold = [0.75, 0.9, 0.95, 0.99]

        parameters = [kernels, bandwidth, threshold]
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

    def predict(self, values: np.ndarray) -> np.ndarray:
        """

        :param values: Werte für die eine Vorhersage getroffen werden soll
        :return: gibt Klassifikationsergebnisse zurück
        """
        scores = np.exp(self.model.score_samples(values))
        self.predictions = [0 if (score > self.probability) else 1 for score in scores ]
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
        Erstellt einen Plot der den KDE darstellt, die Trainings und Testdatenverteilung und die vorhergesagten labels
        Funktioniert nur für den 1D-Fall
        :return:
        """
        fig = go.Figure()
        x0 = self.trainX.reshape(1,-1)[0]
        x1 = self.testX.reshape(1,-1)[0]
        fig.add_trace(go.Scatter(x = x0, y = [0.5 for e in range(len(self.trainX))],
                                 name = "Training data", mode = "markers", marker_color = "blue"))
        fig.add_trace(go.Scatter(x=x1, y=self.testy,
                                 name="test data true", mode="markers", marker_color="red"))
        fig.add_trace(go.Scatter(x=x1, y=self.predictions,
                                 name="test data predicted", mode="markers", marker_color="yellow"))

        min = np.min(self.trainX)
        max = np.max(self.trainX)
        xpredict = np.array(np.arange(min, max, 0.1)).reshape(-1,1)
        dummy ,y = np.exp(self.predict(xpredict))
        x = np.array(np.arange(min, max, 0.1))
        fig.add_trace(go.Scatter(x = x, y = y, marker_color = "orange", name = "Kernel Density Estimator"))
        title = "KDE Recall: " + str(self.recall) + " Precision: " + str(self.precision)\
                + " Probability: " + str(self.probability) + " Kernel: " + str(self.kernel) + " Bandwidth: " \
                + str(self.bandwidth)
        fig.update_layout(title = title)
        fig.show()