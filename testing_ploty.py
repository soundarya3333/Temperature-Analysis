import plotly.graph_objs as go
import plotly.io as pio

# Sample plot
fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
pio.show(fig)
