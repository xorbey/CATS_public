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

def plotRocCurve(model, dataset):
    if dataset == "ParamGood":
        st = "1"
    else:
        st = "2"
    resultlist = list(dbHandlerAnomaly.featcollection.find({"dataset": dataset}))
    resultlist = sorted(resultlist, key=lambda k: k['model'])
    titles = [model['model'] for model in resultlist]
    fig = make_subplots(rows=2, cols=2, x_title='False positive rate',
                        y_title='Recall', subplot_titles=titles)
    i = 0
    j = 0
    x = [e['value'][1] for e in model['roc_table']]
    y = [e['value'][0] for e in model['roc_table']]
    hover = [e['parameter'] for e in model['roc_table']]
    fig.add_trace(go.Scatter(x=x, y=y, name=model['model'], marker_color="Blue", text=hover, showlegend=False,
                             mode="markers"), row=i, col=j)
    fig.update_xaxes(range=[-0.05, 1.05], row=i, col=j)
    fig.update_yaxes(range=[-0.05, 1.05], row=i, col=j)
    fig.update_layout(
        title="ROC-Kurce des Local-Outlier-Factor-Modells",
        template="simple_white",
        font=dict(size=22))
    fig.update_layout(
        template = "simple_white",
        font = dict(size=25))
    fig.show()
    path = "ROCcurveLOF.svg"
    pio.write_image(fig, path, width=1960, height=1080, scale=1)



plotRocCurve("LOF","ParamGood")