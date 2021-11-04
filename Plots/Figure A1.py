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
    resultlist = list(dbHandlerAnomaly.featcollection.find({"dataset": dataset}))
    resultlist = sorted(resultlist, key=lambda k: k['model'])
    titles = [model['model'] for model in resultlist]
    rows = math.ceil(len(titles)/3)
    fig = make_subplots(rows=rows, cols=3, x_title = 'False positive rate',
          y_title = 'True positive rate', subplot_titles = titles)


    i = 0
    j = 1
    n = 1
    for model in resultlist:
        n + 8
        i += 1
        if i == 4:
            i = 1
            j += 1
        x = [e['value'][1] for e in model['roc_table']]
        y = [e['value'][0] for e in model['roc_table']]
        hover = [e['parameter'] for e in model['roc_table']]
        fig.add_trace(go.Scatter(x = x, y = y, name = model['model'], text = hover, showlegend = False, mode = "markers", marker_color = "Blue"), row = j, col = i)
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={'dash': 'dash', 'color': 'grey'}, showlegend=False), row = j, col = i)
        fig.update_xaxes(range=[-0.05, 1.05], row = j, col = i)
        fig.update_yaxes(range=[-0.05, 1.05], row = j, col = i)
    fig.update_layout(
        title="ROC Curves of synthetic data set " + st,
        template = "simple_white",
        font = dict(size=25))
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=25, color='black')
    fig.show()
    path = "G:/Modelle_Paper/Plots/Figures/anomalycomparisonsynthAppend" + st + ".pdf"
    pio.write_image(fig, path, width=1960, height=1080, scale=1)

plotAllRocCurvesAnomaly("ParamGood")