import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = "browser"
import libs.DataBaseHandler as DataBaseHandler
import libs.Helpers as Helpers
import pandas as pd
import math
dbHandlerTrend = DataBaseHandler.DatabaseHandler("mongodb://localhost", 27017, "ModelPaper", "TrendDet")
dbHandlerAnomaly = DataBaseHandler.DatabaseHandler("mongodb://localhost", 27017, "ModelPaper", "AnomalyDet")



def plotAllRocCurvesTrend(dataset):
    if dataset == "ParamGood":
        st = "1"
    else:
        st = "2"
    resultlist = list(dbHandlerTrend.featcollection.find({"dataset": dataset}))
    resultlist = sorted(resultlist, key=lambda k: k['model'])
    titles = [model['model'] for model in resultlist]
    fig = make_subplots(rows=2, cols=2, x_title = 'False positive rate',
          y_title = 'True positive rate', subplot_titles = titles)
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
        fig.add_trace(go.Scatter(x = x, y = y, name = model['model'], marker_color = "Blue", text = hover, showlegend = False, mode = "markers"), row = j, col = i)
        fig.update_xaxes(range=[-0.05, 1.05], row=j, col=i)
        fig.update_yaxes(range=[-0.05, 1.05], row=j, col=i)
    fig.update_layout(
        title="ROC Curves of synthetic data set " + st,
        template = "simple_white",
        font = dict(size=22))
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=22, color='black')
    fig.show()
    path = "G:/Modelle_Paper/Plots/Figures/trendcomparisonsynth" + st + ".pdf"
    pio.write_image(fig, path, width=1960, height=1080, scale=1)

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
          y_title = 'Recall', subplot_titles = titles)


    i = 0
    j = 1
    for model in resultlist:
        i += 1
        if i == 4:
            i = 1
            j += 1
        x = [e['value'][1] for e in model['roc_table']]
        y = [e['value'][0] for e in model['roc_table']]
        hover = [e['parameter'] for e in model['roc_table']]
        fig.add_trace(go.Scatter(x = x, y = y, name = model['model'], text = hover, showlegend = True, mode = "markers"), row = j, col = i)
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={'dash': 'dash', 'color': 'grey'}, showlegend=False), row = j, col = i)
        fig.update_xaxes(range=[-0.05, 1.05], row = j, col = i)
        fig.update_yaxes(range=[-0.05, 1.05], row = j, col = i)
    fig.update_layout(
        title="ROC Curves of synthetic data set " + st,
        template = "simple_white",
        font = dict(size=22))
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=22, color='black')
    fig.show()
    path = "G:/Modelle_Paper/Plots/Figures/anomalycomparisonsynthAppend" + st + ".pdf"
    pio.write_image(fig, path, width=1960, height=1080, scale=1)

def plotRocCurve(model, dataset):
    if dataset == "ParamGood":
        st = "1"
    else:
        st = "2"
    resultlist = list(dbHandlerTrend.featcollection.find({"dataset": dataset}))
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
        title="ROC Curves of synthetic data set " + st,
        template="simple_white",
        font=dict(size=22))
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=22, color='black')
    fig.show()

def plotAUCsTrend(dataset):
    #get all models
    models = dbHandlerTrend.featcollection.distinct("model")
    result = {}
    for model in models:
        auc = Helpers.getAUC(model, dataset, dbHandlerTrend)
        result[model] = auc
    result = pd.Series(result)
    fig = go.Figure(go.Bar(y = result.values, x = result.index))
    fig.show()
    #get auc for models
    #create bar chart
    return



def plotAUCsAnomaly():
    models = dbHandlerAnomaly.featcollection.distinct("model")
    resultGood = {}
    for model in models:
        auc = Helpers.getAUC(model, "ParamGood", dbHandlerAnomaly)
        resultGood[model] = auc
    resultGood = pd.Series(resultGood)
    resultBad = {}
    for model in models:
        auc = Helpers.getAUC(model, "ParamBad", dbHandlerAnomaly)
        resultBad[model] = auc
    resultBad = pd.Series(resultBad)

    fig = go.Figure(go.Bar(y = resultGood.values, x = resultGood.index, name = "Synthetic data set 1"))
    fig.add_trace(go.Bar(y = resultBad.values, x = resultBad.index, name = "Synthetic data set 2"))
    fig.show()
    #get auc for models
    #create bar chart
    return


def plotAllRocCurvesAnomaly(dataset):
    if dataset == "ParamGood":
        st = "1"
    else:
        st = "2"
    fig = go.Figure()
    resultlist = list(dbHandlerAnomaly.featcollection.find({"dataset": dataset}))
    for model in resultlist:
        x = [e['value'][1] for e in model['roc_table']]
        y = [e['value'][0] for e in model['roc_table']]
        hover = [e['parameter'] for e in model['roc_table']]
        fig.add_trace(go.Scatter(x = x, y = y, name = model['model'], text = hover, mode = "markers"))
    fig.add_trace(go.Scatter(x = [0,1], y = [0,1], mode="lines", line={'dash': 'dash', 'color': 'grey'}, showlegend=False))

    fig.update_layout(
        title="ROC curves of synthetic data set " + st,
        template = "simple_white",
        font = dict(size=22),
        xaxis_title = "False positive rate",
        yaxis_title = "True positive rate")
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=22, color='black')
    fig.show()
    path = "G:/Modelle_Paper/Plots/Figures/anomalycomparisonsynth" + st + ".pdf"
    pio.write_image(fig, path, width=1960, height=1080, scale=1)


def getAUCTable():
    models = dbHandlerAnomaly.featcollection.distinct("model")
    resultGood = {}
    for model in models:
        auc = Helpers.getAUC(model, "ParamGood", dbHandlerAnomaly)
        resultGood[model] = auc
    resultGood = pd.Series(resultGood)
    resultBad = {}
    for model in models:
        auc = Helpers.getAUC(model, "ParamBad", dbHandlerAnomaly)
        resultBad[model] = auc
    resultBad = pd.Series(resultBad)
    df = resultGood.to_frame("Synthetic data set 1")
    df["Synthetic data set 2"] = resultBad
    print(df.to_latex())

def getAUCTableAnomaly():
    models = dbHandlerAnomaly.featcollection.distinct("model")
    resultGood = {}
    for model in models:
        auc = Helpers.getAUC(model, "ParamGood", dbHandlerAnomaly)
        resultGood[model] = auc
    resultGood = pd.Series(resultGood)
    resultBad = {}
    for model in models:
        auc = Helpers.getAUC(model, "ParamBad", dbHandlerAnomaly)
        resultBad[model] = auc
    resultBad = pd.Series(resultBad)
    df = resultGood.to_frame("Synthetic data set 1")
    df["Synthetic data set 2"] = resultBad
    print(df.to_latex())

def getAUCTableTrend():
    models = dbHandlerTrend.featcollection.distinct("model")
    resultGood = {}
    for model in models:
        auc = Helpers.getAUC(model, "ParamGood", dbHandlerTrend)
        resultGood[model] = auc
    resultGood = pd.Series(resultGood)
    resultBad = {}
    for model in models:
        auc = Helpers.getAUC(model, "ParamBad", dbHandlerTrend)
        resultBad[model] = auc
    resultBad = pd.Series(resultBad)
    df = resultGood.to_frame("Synthetic data set 1")
    df["Synthetic data set 2"] = resultBad
    print(df.to_latex())