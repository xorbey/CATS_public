import libs.Models.Trend.CrossingAveragesModel as CrossingAveragesModel
import libs.DataBaseHandler as DataBaseHandler
import pandas as pd
import numpy as np
import libs.DataGeneration as DataGen
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"
dbHandler = DataBaseHandler.DatabaseHandler("mongodb://localhost", 27017, "ModelPaper", "TrendDet")
model = "CrossingAvg"
dataset = "ParamGood"

params = DataGen.getParamGood()
paramsNoTrend = DataGen.getParamGoodNoTrend()
timeseries = [{'y': DataGen.getTimeSeries(params[e]), 'trend': True, 'trend_startingpoint': 4368} for e in params]
timeseries.extend([{'y': DataGen.getTimeSeries(paramsNoTrend[e]), 'trend': False, 'trend_startingpoint': None} for e in params])


fig = go.Figure()
for e in timeseries:
    fig.add_trace(go.Scatter(x = [e for e in range(len(e['y']))], y = e['y']))
fig.show()

mk = CrossingAveragesModel.CrossingAveragesModel(timeseries, [], [])
roc_table, trendstart_scores = mk.fit()
roc_table = [{"parameter":str(e), "value":roc_table[e]} for e in roc_table]
trendstart_scores = [{"parameter":str(e), "value":trendstart_scores[e]} for e in trendstart_scores]

exist = len(list(dbHandler.featcollection.find({'model':model, 'dataset':dataset})))

if exist == 1:
    dbHandler.featcollection.update_one( { 'model': model, 'dataset': dataset },{ '$push': { 'roc_table': { '$each': roc_table } } })
    dbHandler.featcollection.update_one( { 'model': model, 'dataset': dataset },{ '$push': { 'trendstart_scores': { '$each': trendstart_scores } } })
else:
    result = [{"model": model, "dataset": dataset, "roc_table": roc_table,
                   "trendstart_scores": trendstart_scores}]
    dbHandler.writeData(result)
