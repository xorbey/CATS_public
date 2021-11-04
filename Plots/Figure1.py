import multiprocessing
from builtins import NotImplementedError
import numpy as np
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
import csv
import os

import libs.Models.ModelParent

import Plots.CrossingAveragesModel as CrossingAveragesModel

data = [2,4,3,6,1,4,2,3,5,2,1,4,3,1,2,3,2,5,6,8,10,7,11,9,12,13,12,16,14,17,14,18,15,16,19,15,20,22,21,19,22]
dfs = []
ydata = data#data['ydata']
trend = 'True'#data['trend'][0]
trend_startingpoint = 16
df = {'y' : ydata, 'trend' : trend, 'trend_startingpoint' : trend_startingpoint, 'datasetname' : "Example data set"}
dfs.append(df)



model = CrossingAveragesModel.CrossingAveragesModel(dfs, [], [])
roc_table, trendstart_scores = model.fit()