
import libs.Models.Anomaly.LSTM as LSTM
import libs.Models.Anomaly.STD as STD
import libs.Models.Anomaly.LocalOutlierFactor as LOF
from plotly.subplots import make_subplots
import libs.DataBaseHandler as DataBaseHandler
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"


featuresSKF = pd.read_csv("G:/HI Paper/Notebooks/zscoresskf.csv")
featuresSKF = featuresSKF['0'][15:].values
featuresSKF = featuresSKF[~np.isnan(featuresSKF)]
fig = make_subplots(rows=1, cols=3, subplot_titles=("LSTM model", "STD Model", "LOF model"))


length = 24

trainX = featuresSKF[:500].reshape(-1,1)
trainX = trainX[~np.isnan(trainX)].reshape(-1,1)

testX = featuresSKF[500:].reshape(-1,1)
testX = testX[~np.isnan(testX)].reshape(-1,1)
anomalies = [559]
anomalies.extend([e for e in range(1538,1553)])
anomalies.extend([e for e in range(1641,1655)])
anomalies.extend([e for e in range(1715,1730)])
anomalies.extend([e for e in range(1873,1888)])
anomalies.extend([e for e in range(1922,1925)])

testy = [0 if e not in anomalies else 1 for e in range(len(testX))]

#TODO add other model here
model = LSTM.LSTMAnomaly(trainX, testX, testy, 50, 0.999, 20)
model.fit()


colormap = {1:"orange", 0:"blue"}
colors = ["blue" for e in range(500)]
colors2 = [colormap[e] for e in model.testyPredicted]
colors.extend(colors2)

colorsXAnomaly = [e for e in range(len(featuresSKF)) if colors[e] == "orange"]
colorsYAnomaly = featuresSKF[colorsXAnomaly]

decision = ["green" for e in range(length)]
for i in range(0, len(colors)-length):
    window = colors[i:i+length]
    if len([e for e in window if e == "orange"]) > 10:
        decision.append("red")
    else:
        decision.append("green")

decisionX = [e for e in range(len(featuresSKF)) if decision[e] == "red"]

fig.add_trace(go.Scatter(x = [e for e in range(len(featuresSKF))], y = featuresSKF, name = "Values", marker_color = "blue", showlegend = False), row = 1, col = 1)
fig.add_trace(go.Scatter(x = colorsXAnomaly, y = colorsYAnomaly,
                           marker_color = "orange", mode = "markers", name = "Predicted anomalies", showlegend = False), row = 1, col = 1)
fig.add_trace(go.Scatter(x = decisionX, y = [20 for e in range(len(decisionX))],
                           marker_color = "red", mode = "markers", name = "Maintenance decision", showlegend = False), row = 1, col = 1)
fig.update_xaxes(title_text="Measurement number", row=1, col=1)
fig.update_yaxes(title_text="Z-score", row=1, col=1)

trainX = featuresSKF[:500].reshape(-1,1)
testX = featuresSKF[500:].reshape(-1,1)
anomalies = [559]
anomalies.extend([e for e in range(1538,1553)])
anomalies.extend([e for e in range(1641,1655)])
anomalies.extend([e for e in range(1715,1730)])
anomalies.extend([e for e in range(1873,1888)])
anomalies.extend([e for e in range(1922,1925)])
testy = [0 if e not in anomalies else 1 for e in range(len(testX))]
#model = LOF.LOFAnomaly(trainX, testX, testy, n_neighbors= 80)
model = STD.STD(trainX, testX, testy,2)
model.fit()

predictions = model.predict(model.testX)

"""fig = go.Figure(go.Scatter(x = [e for e in range(len(testX))], y = [e[0] for e in testX], name = "Values"))
fig.add_trace(go.Scatter(x = [e for e in range(len(testX))], y = scores,  name = "Scores"))
fig.add_trace(go.Scatter(x = [e for e in range(len(testX))], y = predictions,  name = "Predictions"))
fig.show()"""

colormap = {1:"orange", 0:"blue"}
colors = ["blue" for e in range(500)]
colors2 = [colormap[e] for e in predictions]
colors.extend(colors2)

colorsXAnomaly = [e for e in range(len(featuresSKF)) if colors[e] == "orange"]
colorsYAnomaly = featuresSKF[colorsXAnomaly]

length = 24
decision = ["green" for e in range(length)]
for i in range(0, len(colors)-length):
    window = colors[i:i+length]
    if len([e for e in window if e == "orange"]) > 10:
        decision.append("red")
    else:
        decision.append("green")

decisionX = [e for e in range(len(featuresSKF)) if decision[e] == "red"]

fig.add_trace(go.Scatter(x = [e for e in range(len(featuresSKF))], y = featuresSKF, name = "Values", marker_color = "blue", showlegend = True), row = 1, col = 2)
fig.add_trace(go.Scatter(x = colorsXAnomaly, y = colorsYAnomaly,
                           marker_color = "orange", mode = "markers", name = "Predicted trend values", showlegend = True), row = 1, col = 2)
fig.add_trace(go.Scatter(x = decisionX, y = [20 for e in range(len(decisionX))],
                           marker_color = "red", mode = "markers", name = "Maintenance decision", showlegend = True), row = 1, col = 2)
fig.update_xaxes(title_text="Measurement number", row=1, col=2)
fig.update_yaxes(title_text="Z-score", row=1, col=2)

trainX = featuresSKF[:500].reshape(-1,1)
testX = featuresSKF[500:].reshape(-1,1)
anomalies = [559]
anomalies.extend([e for e in range(1538,1553)])
anomalies.extend([e for e in range(1641,1655)])
anomalies.extend([e for e in range(1715,1730)])
anomalies.extend([e for e in range(1873,1888)])
anomalies.extend([e for e in range(1922,1925)])
testy = [0 if e not in anomalies else 1 for e in range(len(testX))]
#model = LOF.LOFAnomaly(trainX, testX, testy, n_neighbors= 80)
model = LOF.LOFAnomaly(trainX, testX, testy,80, 0.01, 3)
model.fit()

predictions, scores = model.predict(model.testX)

"""fig = go.Figure(go.Scatter(x = [e for e in range(len(testX))], y = [e[0] for e in testX], name = "Values"))
fig.add_trace(go.Scatter(x = [e for e in range(len(testX))], y = scores,  name = "Scores"))
fig.add_trace(go.Scatter(x = [e for e in range(len(testX))], y = predictions,  name = "Predictions"))
fig.show()"""

colormap = {1:"orange", 0:"blue"}
colors = ["blue" for e in range(500)]
colors2 = [colormap[e] for e in predictions]
colors.extend(colors2)

colorsXAnomaly = [e for e in range(len(featuresSKF)) if colors[e] == "orange"]
colorsYAnomaly = featuresSKF[colorsXAnomaly]

length = 24
decision = ["green" for e in range(length)]
for i in range(0, len(colors)-length):
    window = colors[i:i+length]
    if len([e for e in window if e == "orange"]) > 10:
        decision.append("red")
    else:
        decision.append("green")

decisionX = [e for e in range(len(featuresSKF)) if decision[e] == "red"]

fig.add_trace(go.Scatter(x = [e for e in range(len(featuresSKF))], y = featuresSKF, name = "Values", marker_color = "blue", showlegend = False), row = 1, col = 3)
fig.add_trace(go.Scatter(x = colorsXAnomaly, y = colorsYAnomaly,
                           marker_color = "orange", mode = "markers", name = "Predicted trend values", showlegend = False), row = 1, col = 3)
fig.add_trace(go.Scatter(x = decisionX, y = [20 for e in range(len(decisionX))],
                           marker_color = "red", mode = "markers", name = "Maintenance decision", showlegend = False), row = 1, col = 3)
fig.update_xaxes(title_text="Measurement number", row=1, col=3)
fig.update_yaxes(title_text="Z-score", row=1, col=3)

fig.update_layout( template="simple_white", title = "Comparison of anomaly detection models for accelerated wear test data"
                   ,font = dict(size=26))
fig.update_layout(legend = dict(font = dict(size = 26, color = "black")))

for i in fig['layout']['annotations']:
    i['font'] = dict(size=26,color='black')
fig.update_layout(legend=dict(font=dict(size=26, color="black")))
fig.update_layout(legend={'itemsizing': 'constant'})
fig.show()
path = "G:/Modelle_Paper/Plots/Figures/anomalycomparisonSKFSTD.pdf"
pio.write_image(fig, path, width=1960, height=1080, scale=1)

