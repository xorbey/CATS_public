import libs.Models.Anomaly.KDESlope as KDESlope
import libs.DataBaseHandler as DataBaseHandler
import libs.Helpers as Helpers
import pandas as pd
import numpy as np
import libs.DataGeneration as DataGen
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"
dbHandler = DataBaseHandler.DatabaseHandler("mongodb://localhost", 27017, "ModelPaper", "AnomalyDet")

#Create data
paramsNoTrend = DataGen.getParamsBadNoTrend()
timeseries = [{'y': DataGen.getTimeSeries(paramsNoTrend[e]), 'anomalies': paramsNoTrend[e]['anomalyposition'], "anomalylengths":  paramsNoTrend[e]['anomalylengthstep'], "anomalytype": paramsNoTrend[e]['anomalytype']} for e in paramsNoTrend]

#Create roc data
rocs = []
distances = []
for series in timeseries:
    seriesy = series['y'].reshape(-1, 1)
    trainX = seriesy[:12*7*24]
    testX = seriesy[12*7*24:]
    if series["anomalytype"] == "collective":
        y = DataGen.collectiveAnomalyLabels(range(len(series['y'])), series["anomalies"], series["anomalylengths"])
    else:
        y = [0 if e not in series['anomalies'] else 1 for e in range(len(seriesy))]
    testy = y[12*7*24:]
    model = KDESlope.KDESlope(trainX, testX, testy)
    roc, distance = model.getROC()
    rocs.append(roc)
    distances.append(distance)

#Average Roc and distance data
result = []
rocAvgwithKey, distancesAvgwithKey = Helpers.getMeanData(rocs, distances)

result.append({"model": "KDESlope", "dataset": "ParamGood", "roc_table": rocAvgwithKey,
               "trendstart_scores":distancesAvgwithKey})

dbHandler.writeData(result)
