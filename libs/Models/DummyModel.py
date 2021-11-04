from builtins import NotImplementedError

import numpy as np

from libs.Models.ModelParent import ModelParent


class DummyModel(ModelParent):
    def __init__(self, trainX: np.array, testX: np.array, testy: np.array):
        super().__init__(trainX, testX, testy)

        self.model = None #Save the object which will be trained/ used for predictions in here

    def fit(self) -> None:
        """
        Put code to train model here
        """

        raise NotImplementedError

    def predict(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def showResults(self):
        """
        Put code to explain the model here
        """

        raise NotImplementedError
