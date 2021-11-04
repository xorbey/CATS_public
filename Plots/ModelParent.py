from builtins import NotImplementedError
import numpy as np
#from lib.DataStructureComponents.Models.ModelParent import ModelParent
from abc import ABC, abstractmethod
from copy import copy
from typing import Optional
from typing import List
import pandas as pd
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import pymannkendall as mk
import skfuzzy as fuzz
import math

class ModelParent(ABC):
    def __init__(self, trainDfs: List[pd.DataFrame], testX: np.ndarray, testy: np.ndarray):
        self.trainDfs: List[pd.DataFrame] = trainDfs
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
