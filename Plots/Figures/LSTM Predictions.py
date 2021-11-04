
import libs.Models.Anomaly.LSTM as LSTM
import libs.DataBaseHandler as DataBaseHandler
import libs.Helpers as Helpers
import libs.DataGeneration as DataGen
import plotly.io as pio
pio.renderers.default = "browser"
dbHandler = DataBaseHandler.DatabaseHandler("mongodb://localhost", 27017, "ModelPaper", "AnomalyDet")
model = "LSTM"
dataset = "ParamGood"
#Create data
paramsNoTrend = DataGen.getParamGoodNoTrend()
timeseries = [{'y': DataGen.getTimeSeries(paramsNoTrend[e]), 'anomalies': paramsNoTrend[e]['anomalyposition'], "anomalylengths":  paramsNoTrend[e]['anomalylengthstep'], "anomalytype": paramsNoTrend[e]['anomalytype']} for e in paramsNoTrend]


series = timeseries[0]
seriesy = series['y'].reshape(-1, 1)
trainX = seriesy[:1 * 7 * 24]
testX = seriesy[12 * 7 * 24:]
if series["anomalytype"] == "collective":
    y = DataGen.collectiveAnomalyLabels(range(len(series['y'])), series["anomalies"], series["anomalylengths"])
else:
    y = [0 if e not in series['anomalies'] else 1 for e in range(len(seriesy))]
testy = y[12 * 7 * 24:]
model = LSTM.LSTMAnomaly(trainX, testX, testy, 50, 0.9999)
model.fit()

train_size = int(len(model.trainX))
XX, Xy = model.create_Xy_dataset(model.concatX, model.split)
trainXX, testXX = XX[:train_size - model.split], XX[train_size - model.split:]
trainXy, testXy = Xy[:train_size - model.split], Xy[train_size - model.split:]
input_shape = (model.split, 1)
trainXX = trainXX.reshape((trainXX.shape[0], trainXX.shape[1], 1))
testXX = testXX.reshape((testXX.shape[0], testXX.shape[1], 1))
predictions = model.model.predict(testXX)

import plotly.graph_objs as go
fig = go.Figure(go.Scatter(x = [e for e in range(len(testX))], y = [e[0] for e in testX], name = "Values"))
fig.add_trace(go.Scatter(x = [e for e in range(len(testX))], y = [e[0] for e in predictions], name = "Predictions"))
import plotly.io as pio
pio.renderers.default = "browser"

fig.update_layout(xaxis_title = "Measurement number", yaxis_title = "Value")
fig.show()