"""
To run this script, the results for the ROC curves must be already stored in the MongoDB.
To create these results, run all scripts in the Code directory

"""

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = "browser"
import libs.DataBaseHandler as DataBaseHandler
import libs.Helpers as Helpers
import pandas as pd
import math
from plotly.validators.scatter.marker import SymbolValidator

raw_symbols = SymbolValidator().values

dbHandlerAnomaly = DataBaseHandler.DatabaseHandler("mongodb://localhost", 27017, "ModelPaper", "AnomalyDet")


def plotAllRocCurvesAnomaly(dataset):
    if dataset == "ParamGood":
        st = "1"
    else:
        st = "2"
    fig = go.Figure()
    resultlist = list(dbHandlerAnomaly.featcollection.find({"dataset": dataset}))
    i = 1
    for model in resultlist:
        i = i + 8
        x = [e['value'][1] for e in model['roc_table']]
        y = [e['value'][0] for e in model['roc_table']]
        hover = [e['parameter'] for e in model['roc_table']]
        fig.add_trace(go.Scatter(x = x, y = y, name = model['model'],  marker = dict(size = 18, opacity=0.5), text = hover, mode = "markers", marker_symbol = raw_symbols[i]))
    fig.add_trace(go.Scatter(x = [0,1], y = [0,1], mode="lines", line={'dash': 'dash', 'color': 'grey'}, showlegend=False))

    fig.update_layout(
        title="ROC curves of synthetic data set " + st,
        template = "simple_white",
        font = dict(size=26),
        xaxis_title = "False positive rate",
        yaxis_title = "True positive rate")
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=26, color='black')
    fig.show()
    path = "G:/Modelle_Paper/Plots/Figures/anomalycomparisonsynth" + st + ".pdf"
    pio.write_image(fig, path, width=1960, height=1080, scale=1)


plotAllRocCurvesAnomaly("ParamGood")



