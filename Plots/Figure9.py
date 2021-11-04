
import libs.Models.Trend.CoxStuartNew as CoxStuart
import libs.Models.Trend.MannKendallModel as MannKendall
import libs.Models.Trend.CrossingAveragesModel as CrossingAveragesModel
import libs.DataBaseHandler as DataBaseHandler
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import  make_subplots
import plotly.io as pio
pio.renderers.default = "browser"

features = pd.HDFStore("G:/InstationaryVelFeatures/features_instationary.hdf")["features"]
featuresSKF = pd.read_csv("G:/HI Paper/Notebooks/zscoresskf.csv")
featuresSKF = featuresSKF['0'][15:].values
featuresBMW = features.Zscore.values
length = 168*3

fig = make_subplots(rows=1, cols=2, subplot_titles=("Mann Kendall Model", "Cox Stuart Model"))

mk = MannKendall.MannKendallModel(featuresSKF, [], [])

steps = range(0, len(featuresSKF)-length)

colormap = {True:"orange", False:"blue"}
colors = ["blue" for e in range(length)]
for i in steps:
    start = i
    if start + length >= len(featuresSKF)-1:
        stop = len(featuresSKF)-1
    else:
        stop = start + length
    window = featuresSKF[start:stop]
    colors.append(colormap[mk.predict(window, 0.006)])

colorsXAnomaly = [e for e in range(len(featuresSKF)) if colors[e] == "orange"]
colorsYAnomaly = featuresSKF[colorsXAnomaly]

decision = ["green" for e in range(length)]
for i in range(0, len(colors)-length):
    window = colors[i:i+length]
    if len([e for e in window if e == "orange"]) > length/2:
        decision.append("yellow")
    else:
        decision.append("green")

decisionX = [e for e in range(len(featuresSKF)) if decision[e] == "yellow"]

fig.add_trace(go.Scatter(x = [e for e in range(len(featuresSKF))], y = featuresSKF,  name = "Values", marker_color = "blue"), row = 1, col = 1)
fig.add_trace(go.Scatter(x = colorsXAnomaly, y = colorsYAnomaly,
                           marker_color = "orange", mode = "markers",  name = "Predicted trend values"),row = 1, col = 1)
fig.add_trace(go.Scatter(x = decisionX, y = [20 for e in range(len(decisionX))],
                           marker_color = "yellow", mode = "markers", name = "Maintenance decision", showlegend = True), row = 1, col = 1)
fig.add_trace(go.Scatter(x = [e for e in range(len(pd.Series(featuresSKF).rolling(length).mean()))], y = pd.Series(featuresSKF).rolling(length).mean(),  name = "Rolling mean", marker_color = "red"), row = 1, col = 1)

fig.update_xaxes(title_text="Measurement number", row=1, col=1)
fig.update_yaxes(title_text="Z-score", row=1, col=1)

mk = CoxStuart.CoxStuart(featuresSKF, [], [])


steps = range(0, len(featuresSKF)-length)

colormap = {True:"orange", False:"blue"}
colors = ["blue" for e in range(length)]
for i in steps:
    start = i
    if start + length >= len(featuresSKF)-1:
        stop = len(featuresSKF)-1
    else:
        stop = start + length
    window = featuresSKF[start:stop]
    colors.append(colormap[mk.predict(window, 0.006)])

colorsXAnomaly = [e for e in range(len(featuresSKF)) if colors[e] == "orange"]
colorsYAnomaly = featuresSKF[colorsXAnomaly]

decision = ["green" for e in range(length)]
for i in range(0, len(colors)-length):
    window = colors[i:i+length]
    if len([e for e in window if e == "orange"]) > length/2:
        decision.append("yellow")
    else:
        decision.append("green")

decisionX = [e for e in range(len(featuresSKF)) if decision[e] == "yellow"]

fig.add_trace(go.Scatter(x = [e for e in range(len(featuresSKF))], y = featuresSKF, name = "Values", marker_color = "blue", showlegend = False), row = 1, col = 2)
fig.add_trace(go.Scatter(x = colorsXAnomaly, y = colorsYAnomaly,
                           marker_color = "orange", mode = "markers", name = "Predicted trend values", showlegend = False), row = 1, col = 2)
fig.add_trace(go.Scatter(x = decisionX, y = [20 for e in range(len(decisionX))],
                           marker_color = "yellow", mode = "markers", name = "Maintenance decision", showlegend = False), row = 1, col = 2)
fig.add_trace(go.Scatter(x = [e for e in range(len(pd.Series(featuresSKF).rolling(length).mean()))],showlegend = False, y = pd.Series(featuresSKF).rolling(length).mean(),  name = "Rolling mean", marker_color = "red"), row = 1, col = 2)

fig.update_xaxes(title_text="Measurement number", row=1, col=2)
fig.update_yaxes(title_text="Z-score", row=1, col=2)



fig.update_layout( template="simple_white", title = "Comparison of trend detection models for accelerated wear test data"
                   ,font = dict(size=22))
for i in fig['layout']['annotations']:
    i['font'] = dict(size=26,color='black')
fig.update_layout(legend = dict(font = dict(size = 26, color = "black")))
fig.update_layout(legend= {'itemsizing': 'constant'})
fig.show()
path = "G:/Modelle_Paper/Plots/Figures/trendcomparisonskf.pdf"
pio.write_image(fig, path, width=1960, height=1080, scale=1)

