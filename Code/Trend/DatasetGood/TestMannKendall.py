

import libs.Models.Trend.MannKendallModel as MannKendall
import libs.DataBaseHandler as DataBaseHandler
import pandas as pd
import numpy as np
import libs.DataGeneration as DataGen
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"
dbHandler = DataBaseHandler.DatabaseHandler("mongodb://localhost", 27017, "ModelPaper", "TrendDet")
model = "MannKendall"
dataset = "ParamGood"


params = DataGen.getParamGood()
paramsNoTrend = DataGen.getParamGoodNoTrend()
timeseries = [{'y': DataGen.getTimeSeries(params[e]), 'trend': True, 'trend_startingpoint': 4368} for e in params]
timeseries.extend([{'y': DataGen.getTimeSeries(paramsNoTrend[e]), 'trend': False, 'trend_startingpoint': None} for e in params])

fig = go.Figure()
for e in timeseries:
    fig.add_trace(go.Scatter(x = [e for e in range(len(e['y']))], y = e['y']))
fig.show()

mk = MannKendall.MannKendallModel(timeseries, [], [])
roc_table, trendstart_scores = mk.fit()



roc_table = [{"parameter":float(e), "value":roc_table[e]} for e in roc_table]
trendstart_scores = [{"parameter":float(e), "value":trendstart_scores[e]} for e in trendstart_scores]
result = []
result.append({"model": "MannKendall", "dataset": "ParamGood", "roc_table": roc_table,
               "trendstart_scores":trendstart_scores})

exist = len(list(dbHandler.featcollection.find({'model':model, 'dataset':dataset})))

if exist == 1:
    dbHandler.featcollection.update_one( { 'model': model, 'dataset': dataset },{ '$push': { 'roc_table': { '$each': roc_table } } })
    dbHandler.featcollection.update_one( { 'model': model, 'dataset': dataset },{ '$push': { 'trendstart_scores': { '$each': trendstart_scores } } })
else:
    result = [{"model": model, "dataset": dataset, "roc_table": roc_table,
                   "trendstart_scores": trendstart_scores}]
    dbHandler.writeData(result)