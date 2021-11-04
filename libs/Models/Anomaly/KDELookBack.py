import libs.Models.Anomaly.KernelDensityEstimator as KDE
import libs.DataBaseHandler as DataBaseHandler
import libs.Helpers as Helpers
import libs.DataGeneration as DataGen
import plotly.io as pio
from sklearn.neighbors import KernelDensity
import numpy as np
from plotly.subplots import make_subplots




pio.renderers.default = "browser"
dbHandler = DataBaseHandler.DatabaseHandler("mongodb://localhost", 27017, "ModelPaper", "AnomalyDet")
model = "KDE"
dataset = "ParamGood"

#Create data
paramsNoTrend = DataGen.getParamsGoodNoTrend()
timeseries = [{'y': DataGen.getTimeSeries(paramsNoTrend[e]), 'anomalies': paramsNoTrend[e]['anomalyposition'], "anomalylengths":  paramsNoTrend[e]['anomalylengthstep'], "anomalytype": paramsNoTrend[e]['anomalytype']} for e in paramsNoTrend]

series = timeseries[0]
seriesy = series['y'].reshape(-1, 1)
trainX = seriesy[:12*7*24]
testX = seriesy[12*7*24:]
if series["anomalytype"] == "collective":
    y = DataGen.collectiveAnomalyLabels(range(len(series['y'])), series["anomalies"], series["anomalylengths"])
else:
    y = [0 if e not in series['anomalies'] else 1 for e in range(len(seriesy))]
testy = y[12*7*24:]

length = 48
lengthArray = len(trainX)

chunks = []
for e in range(int(lengthArray-length -1)):
    if e + length < lengthArray-1:
        chunks.append(np.array([e[0] for e in trainX[e:e + length]]))
    else:
        break

model = KernelDensity(kernel = "gaussian", bandwidth = 0.5).fit(chunks)

lengthArray = len(testX)
testChunks = []
for e in range(int(lengthArray-length -1)):
    if e + length < lengthArray-1:
        testChunks.append(np.array([e[0] for e in testX[e:e + length]]).reshape(1,-1))
    else:
        break

predictions = [np.exp(model.score(e)*1000) for e in testChunks]
labels = [0 for e in range(len(testX) - len(predictions))]
labels = np.append(labels, predictions)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x = [e for e in range(len(testX))], y = [e[0] for e in testX]), secondary_y = "False")
fig.add_trace(go.Scatter(x = [e for e in range(len(testX))], y = labels), secondary_y = "True")


fig.show()

