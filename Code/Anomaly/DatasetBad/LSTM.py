import libs.Models.Anomaly.LSTM as LSTM
import libs.DataBaseHandler as DataBaseHandler
import libs.Helpers as Helpers

import libs.DataGeneration as DataGen

import plotly.io as pio

pio.renderers.default = "browser"
dbHandler = DataBaseHandler.DatabaseHandler("mongodb://localhost", 27017, "ModelPaper", "AnomalyDet")
modelName = "LSTM"
dataset = "ParamBad"

#Create data
paramsNoTrend = DataGen.getParamsBadNoTrend()
timeseries = [{'y': DataGen.getTimeSeries(paramsNoTrend[e]), 'anomalies': paramsNoTrend[e]['anomalyposition'], "anomalylengths":  paramsNoTrend[e]['anomalylengthstep'], "anomalytype": paramsNoTrend[e]['anomalytype']} for e in paramsNoTrend]

rocs = []
distances = []
i = 0
for series in timeseries:
    print(i)
    i += 1
    seriesy = series['y'].reshape(-1, 1)
    trainX = seriesy[:1*7*24]
    testX = seriesy[12*7*24:]
    if series["anomalytype"] == "collective":
        y = DataGen.collectiveAnomalyLabels(range(len(series['y'])), series["anomalies"], series["anomalylengths"])
    else:
        y = [0 if e not in series['anomalies'] else 1 for e in range(len(seriesy))]
    testy = y[12*7*24:]
    model = LSTM.LSTMAnomaly(trainX, testX, testy)
    roc, distance = model.getROC()
    rocs.append(roc)
    distances.append(distance)

#Average Roc and distance data

rocAvgwithKey, distancesAvgwithKey = Helpers.getMeanData(rocs, distances)

#Add data to database
exist = len(list(dbHandler.featcollection.find({'model':modelName, 'dataset':dataset})))

if exist == 1:
    dbHandler.featcollection.update_one( { 'model': modelName, 'dataset': dataset },{ '$push': { 'roc_table': { '$each': rocAvgwithKey } } })
    dbHandler.featcollection.update_one( { 'model': modelName, 'dataset': dataset },{ '$push': { 'trendstart_scores': { '$each': distancesAvgwithKey } } })
else:
    result = [{"model": modelName, "dataset": dataset, "roc_table": rocAvgwithKey,
                   "trendstart_scores": distancesAvgwithKey}]
    dbHandler.writeData(result)