import libs.Models.Anomaly.IsolationForest as IsolationForest
import libs.DataBaseHandler as DataBaseHandler
import libs.Helpers as Helpers
import libs.DataGeneration as DataGen
import plotly.io as pio

pio.renderers.default = "browser"
dbHandler = DataBaseHandler.DatabaseHandler("mongodb://localhost", 27017, "ModelPaper", "AnomalyDet")
model = "Isolation Forest"
dataset = "ParamGood"

#Create data
paramsNoTrend = DataGen.getParamsGoodNoTrend()
timeseries = [{'y': DataGen.getTimeSeries(paramsNoTrend[e]), 'anomalies': paramsNoTrend[e]['anomalyposition'], "anomalylengths":  paramsNoTrend[e]['anomalylengthstep'], "anomalytype": paramsNoTrend[e]['anomalytype']} for e in paramsNoTrend]

rocs = []
distances = []

series = timeseries['y']
steps = 5

X = []
for i in range(len(series)):
    # find the end of this pattern
    end_ix = i + steps
    # check if we are beyond the sequence
    if end_ix > len(series) - 1:
        break
    # gather input and output parts of the pattern
    seq_x, seq_y = series[i:end_ix], series[end_ix]
    X.append(seq_x)


model = IsolationForest(n_estimators=100, contamination=0.01).fit(X)


