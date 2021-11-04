import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import precision_score, recall_score
import plotly.graph_objects as go
from libs.Models.ModelParent import ModelParent
import itertools


class GPC(ModelParent):
    def __init__(self, trainX: np.array, testX: np.array, testy: np.array, kernel=RBF(10, (1e-3, 1e-1)),
                 threshold=0.3, split = 5):
        """
        :param trainX: Training data
        :param testX: Test data
        :param testy: Labels of the test data
        :param kernel: Kernel for Gaussian Process regression
        :param threshold: Anomaly threshold in range [0,1]
        """
        super().__init__(trainX, testX, testy)
        self.threshold = threshold
        self.kernel = kernel
        self.model = None  # Save the object which will be trained/ used for predictions in here
        self.precision = 0
        self.recall = 0
        self.predictions = []
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
        Fits model with training data. Calculates precision and recall of the model.
        """
        self.standardize_dataset()
        train_size = int(len(self.trainX))
        XX, Xy = self.create_Xy_dataset(np.concatenate([self.trainX, self.testX], axis=0), self.split)
        #XX, Xy = self.create_Xy_dataset(self.concatX, self.split)
        trainXX, testXX = XX[:train_size - self.split], XX[train_size - self.split:]
        trainXy, testXy = Xy[:train_size - self.split], Xy[train_size - self.split:]
        input_shape = (self.split, 1)
        trainXX = trainXX.reshape(trainXX.shape[0], trainXX.shape[1])
        testXX = testXX.reshape(testXX.shape[0], testXX.shape[1])

        y_traindummy = np.ones(len(trainXX)) #Dummy training labels for GP regressor
        self.model = GaussianProcessRegressor(kernel=self.kernel).fit(trainXX, y_traindummy)
        self.testyPredicted = self.predict(testXX)[0]
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
        threshold = [0.7, 0.8, 0.9, 0.95]
        kernellen = [1]
        kernellowerbound = [1e-5]
        kernelupperbound = [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.013, 0.021, 0.034, 0.055, 0.089, 0.144,
                            0.233, 0.377,
                            0.610, 0.987]  # fibonacci like distribution
        parameters = [threshold, kernellen, kernellowerbound, kernelupperbound]
        parameters = list(itertools.product(*parameters))
        roc = []
        distances = []
        for e in parameters:
            self.threshold = e[0]
            self.kernel = RBF(e[1], (e[2], e[3]))
            self.fit()
            roc.append({"parameter": e,"value": [float(self.recall), float(1- self.specificity)]})
            distances.append({"parameter": e, "value": float(self.getStartDeltas())})
        return roc, distances

    def predict(self, values: np.array) -> np.ndarray:
        """
        Calculates anomaly scores of the test data based on the training data.
        :param values: Values to determine anomaly score on.
        :return: Returns binary anomaly predictions and raw anomaly scores
        """
        scores = self.model.predict(values) - self.threshold
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
        Plots the GP Decision Function
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
        xpredict = np.array(np.arange(min, max, 0.1)).reshape(-1, 1)
        y = self.predict(xpredict)[1]
        x = np.array(np.arange(min, max, 0.1))
        fig.add_trace(go.Scatter(x=x, y=y, marker_color="orange", name="GPC Decision function"))
        title = "GPC Recall: " + str(self.recall) + " Precision: " + str(self.precision) + "\n" + "Kernel: " \
                + str(self.kernel) + " Threshold: " + str(self.threshold)
        fig.update_layout(title=title)
        fig.show()
