import libs.DataGeneration as DataGen
import numpy as np

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

params = DataGen.getParamGood()

paramset = params[12]

testrange =np.array([e for e in range(0,24*7*52)])
trend = DataGen.lin(testrange, 0.0005)
seas = DataGen.productioncycle([1 for e in testrange], paramset["seasoningfactorstep"])
noise = DataGen.uniformNoise(testrange, paramset["noiseamp"])
anomaly = DataGen.collectiveAnomaly(testrange,[1100, 1425, 1738],
                                   paramset["anomalylengthstep"], paramset["anomalymagnitude"])

y = trend + seas + noise + anomaly

fig = make_subplots( vertical_spacing=0.15,rows=3, cols=2, subplot_titles=("Trend", "Seasonality", "Noise", "Anomaly", "Superpositioned time series"))

fig.add_trace(go.Scatter(x = testrange[1000:2000], y = trend[1000:2000], name = "trend", showlegend = False, marker_color = "Blue"), row = 1, col = 1)
fig.add_trace(go.Scatter(x = testrange[1000:2000], y = seas[1000:2000], name = "seas", showlegend = False, marker_color = "Blue"), row = 1, col = 2)
fig.add_trace(go.Scatter(x = testrange[1000:2000], y = noise[1000:2000], name = "noise", showlegend = False, marker_color = "Blue"), row = 2, col = 1)
fig.add_trace(go.Scatter(x = testrange[1000:2000], y = anomaly[1000:2000], name = "anomaly", showlegend = False, marker_color = "Blue"), row = 2, col = 2)
fig.add_trace(go.Scatter(x = testrange[1000:2000], y = y[1000:2000], name = "anomaly", showlegend = False, marker_color = "Blue"), row = 3, col = 1)


fig.update_layout( template="simple_white", title = "Synthetic data composition example"
                   ,font = dict(size=20))
fig['layout']['yaxis']['position'] = 0
for i in fig['layout']['annotations']:
    i['font'] = dict(size=20,color='black')

fig.update_xaxes(title_text="Time step", row=1, col=1)
fig.update_yaxes(title_text="Value", row=1, col=1)
fig.update_xaxes(title_text="Time step", row=1, col=2)
fig.update_yaxes(title_text="Value", row=1, col=2)
fig.update_xaxes(title_text="Time step", row=2, col=1)
fig.update_yaxes(title_text="Value", row=2, col=1)
fig.update_xaxes(title_text="Time step", row=2, col=2)
fig.update_yaxes(title_text="Value", row=2, col=2)
fig.update_xaxes(title_text="Time step", row=3, col=1)
fig.update_yaxes(title_text="Value", row=3, col=1)


fig.show()

#fig.write_image("G:/Modelle_Paper/Plots/Figures/superpositionedtimeseries.pdf")
path = "G:/Modelle_Paper/Plots/Figures/superpositionedtimeseries.pdf"
pio.write_image(fig, path, width=1960, height=1080, scale=1)

