from abc import ABC, abstractmethod
from copy import copy
from typing import Optional

import numpy as np


class ModelParent(ABC):
    def __init__(self, trainX: np.ndarray, testX: np.ndarray, testy: np.ndarray):
        self.trainX: np.ndarray = trainX
        self.testX: np.ndarray = testX
        self.testy: np.ndarray = testy

        self.model: Optional[np.ndarray] = None

    @abstractmethod
    def fit(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __getstate__(self):
        state = copy(self.__dict__)

        state = state['model']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
