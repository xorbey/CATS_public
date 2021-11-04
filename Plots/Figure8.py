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

dbHandlerTrend = DataBaseHandler.DatabaseHandler("mongodb://localhost", 27017, "ModelPaper", "TrendDet")


def plotAllRocCurvesTrend(dataset):
    if dataset == "ParamGood":
        st = "1"
    else:
        st = "2"
    resultlist = list(dbHandlerTrend.featcollection.find({"dataset": dataset}))
    resultlist = sorted(resultlist, key=lambda k: k['model'])
    titles = [model['model'] for model in resultlist]
    fig = make_subplots(rows=2, cols=2, x_title = 'False positive rate',
          y_title = 'Recall', subplot_titles = titles)
    i = 0
    j = 1
    for model in resultlist:
        i += 1
        print("row:" + str(j) + " col:" + str(i))
        if i == 3:
            i = 1
            j = 2
        x = [e['value'][1] for e in model['roc_table']]
        y = [e['value'][0] for e in model['roc_table']]
        hover = [e['parameter'] for e in model['roc_table']]
        fig.add_trace(go.Scatter(x = x, y = y, name = model['model'],marker =dict(size=18), marker_color = "Blue", text = hover, showlegend = False, mode = "markers"), row = j, col = i)
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={'dash': 'dash', 'color': 'grey'}, showlegend=False), row = j, col = i)
        fig.update_xaxes(range=[-0.05, 1.05], row=j, col=i)
        fig.update_yaxes(range=[-0.05, 1.05], row=j, col=i)
    fig.update_layout(
        title="ROC curves of synthetic data set " + st,
        template = "simple_white",
        font = dict(size=26))
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=26, color='black')
    fig.show()
    path = "G:/Modelle_Paper/Plots/Figures/trendcomparisonsynth" + st + ".pdf"
    pio.write_image(fig, path, width=1960, height=1080, scale=1)


plotAllRocCurvesTrend("ParamBad")


