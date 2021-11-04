import libs.Models.Anomaly.CNN as CNN
import libs.DataBaseHandler as DataBaseHandler
import libs.Helpers as Helpers
import libs.DataGeneration as DataGen
import plotly.io as pio
import plotly.graph_objs as go

pio.renderers.default = "browser"

#Create data
paramsNoTrend = DataGen.getParamsBadNoTrend()
timeseries = [{'y': DataGen.getTimeSeries(paramsNoTrend[e]), 'anomalies': paramsNoTrend[e]['anomalyposition'], "anomalylengths":  paramsNoTrend[e]['anomalylengthstep'], "anomalytype": paramsNoTrend[e]['anomalytype']} for e in paramsNoTrend]


series = timeseries[0]['y']
y = [1 if e in timeseries[0]['anomalies'] else 0 for e in range(len(series[2100:2610]))]

colormap = {1:"orange", 0:"blue"}
colors = []
colors2 = [colormap[e] for e in y]
colors.extend(colors2)


fig = go.Figure(go.Scatter(x = [e for e in range(2100,2610)], marker = dict(opacity = 0.5), y= series[2100:2610], name = "Values", marker_color = "blue"))
fig.add_trace(go.Scatter(mode = "markers", marker = dict(size = 18), marker_color = "orange", x = [e for e in timeseries[0]['anomalies'] if (e > 2100) & (e < 2600)], y = series[[e for e in timeseries[0]['anomalies'] if (e > 2100) & (e < 2600)]], name = "Anomalies"))
fig.update_layout( template="simple_white", title = "Synthetic data composition example"
                   ,font = dict(size=26), xaxis_title = "Time step", yaxis_title = "Value")
fig.show()

path = "G:/Modelle_Paper/Plots/Figures/noisytimeseries.pdf"
pio.write_image(fig, path, width=1960, height=1080, scale=1)