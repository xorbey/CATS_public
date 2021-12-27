import libs.Models.Anomaly.LSTM as LSTM
import libs.Models.Anomaly.LocalOutlierFactor as LOF
from plotly.subplots import make_subplots
import libs.DataBaseHandler as DataBaseHandler
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

features = pd.HDFStore("G:/Vibration Signals Test/TUMzscores.hdf")["Vib"]
featuresBMW = features.values
featuresBMW = featuresBMW[~np.isnan(featuresBMW)]
fig = make_subplots(rows=1, cols=2, subplot_titles=("LSTM model", "LOF model"))



length = 24

trainX = featuresBMW[:200].reshape(-1,1)
testX = featuresBMW[200:].reshape(-1,1)

anomalies = [50,52,57, 252, 263, 279, 486]
anomalies.extend([e for e in range(504,599)])
anomalies.extend([e for e in range(616 ,664)])
anomalies.extend([e for e in range(709,717)])

testy = [0 if e not in anomalies else 1 for e in range(len(testX))]

#TODO add other model here
model = LSTM.LSTMAnomaly(trainX, testX, testy, 50, 0.9999)
model.fit()


colormap = {1:"orange", 0:"blue"}
colors = ["blue" for e in range(200)]
colors2 = [colormap[e] for e in model.testyPredicted]
colors.extend(colors2)

colorsXAnomaly = [e for e in range(len(featuresBMW)) if colors[e] == "orange"]
colorsYAnomaly = featuresBMW[colorsXAnomaly]

decision = ["green" for e in range(length)]
for i in range(0, len(colors)-length):
    window = colors[i:i+length]
    if len([e for e in window if e == "orange"]) > 10:
        decision.append("red")
    else:
        decision.append("green")

decisionX = [e for e in range(len(featuresBMW)) if decision[e] == "red"]


fig.add_trace(go.Scatter(x = [e for e in range(len(featuresBMW))], y = featuresBMW, name = "Values", marker_color = "blue", showlegend = False), row = 1, col = 1)
fig.add_trace(go.Scatter(x = colorsXAnomaly, y = colorsYAnomaly,
                           marker_color = "orange", mode = "markers", name = "Predicted anomalies", showlegend = False), row = 1, col = 1)
fig.add_trace(go.Scatter(x = decisionX, y = [20 for e in range(len(decisionX))],
                           marker_color = "red", mode = "markers", name = "Maintenance decision", showlegend = False), row = 1, col = 1)
fig.update_xaxes(title_text="Measurement number", row=1, col=1)
fig.update_yaxes(title_text="Z-score", row=1, col=1)


trainX = featuresBMW[:200].reshape(-1,1)
trainX = trainX[~np.isnan(trainX)].reshape(-1,1)

testX = featuresBMW[200:].reshape(-1,1)
testX = testX[~np.isnan(testX)].reshape(-1,1)
anomalies = [50,52,57, 252, 263, 279, 486]
anomalies.extend([e for e in range(504,599)])
anomalies.extend([e for e in range(616 ,664)])
anomalies.extend([e for e in range(709,717)])

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
colors = ["blue" for e in range(200)]
colors2 = [colormap[e] for e in model.testyPredicted]
colors.extend(colors2)

colorsXAnomaly = [e for e in range(len(featuresBMW)) if colors[e] == "orange"]
colorsYAnomaly = featuresBMW[colorsXAnomaly]

decision = ["green" for e in range(length)]
for i in range(0, len(colors)-length):
    window = colors[i:i+length]
    if len([e for e in window if e == "orange"]) > 10:
        decision.append("red")
    else:
        decision.append("green")

decisionX = [e for e in range(len(featuresBMW)) if decision[e] == "red"]

fig.add_trace(go.Scatter(x = [e for e in range(len(featuresBMW))], y = featuresBMW, name = "Values", marker_color = "blue", showlegend = True), row = 1, col = 2)
fig.add_trace(go.Scatter(x = colorsXAnomaly, y = colorsYAnomaly,
                           marker_color = "orange", mode = "markers", name = "Predicted anomalies", showlegend = True), row = 1, col = 2)
fig.add_trace(go.Scatter(x = decisionX, y = [20 for e in range(len(decisionX))],
                           marker_color = "red", mode = "markers", name = "Maintenance decision", showlegend = True), row = 1, col = 2)
fig.update_xaxes(title_text="Measurement number", row=1, col=2)
fig.update_yaxes(title_text="Z-score", row=1, col=2)



fig.update_layout( template="simple_white", title = "Comparison of anomaly detection models for accelerated wear test data"
                   ,font = dict(size=26))


for i in fig['layout']['annotations']:
    i['font'] = dict(size=26,color='black')
fig.update_layout(legend = dict(font = dict(size = 26, color = "black")))
fig.update_layout(legend= {'itemsizing': 'constant'})

fig.show()
path = "G:/Modelle_Paper/Plots/Figures/anomalycomparisonBMW.pdf"
#pio.write_image(fig, path, width=1960, height=1080, scale=1)

