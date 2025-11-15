# EN: Dash application to explore pairwise Iris dataset features
# ES: Aplicación Dash para explorar pares de características del dataset Iris

# EN: Import required libraries
# ES: Importar las librerías necesarias
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.datasets import load_iris
import pandas as pd

# EN: Load Iris dataset and prepare DataFrame
# ES: Cargar el dataset Iris y preparar el DataFrame
iris = load_iris(as_frame=True)
df = iris.frame
df['species'] = iris.target_names[iris.target]

# EN: Select only numeric columns for plotting
# ES: Seleccionar solo las columnas numéricas para los gráficos
num_cols = df.select_dtypes(include='number').columns.to_list()

# EN: Initialize Dash application
# ES: Inicializar la aplicación Dash
app = dash.Dash(__name__)

# EN: Define layout with a slider and a graph
# ES: Definir el layout con un control deslizante y un gráfico
app.layout = html.Div([
    html.H4("Iris feature-pair explorer"),
    dcc.Slider(
        id='slider',
        min=0,
        max=len(num_cols)-2,          # last index -1 would leave no y-axis
        step=1,
        value=0,
        marks={i: num_cols[i] for i in range(len(num_cols))}
    ),
    dcc.Graph(id='live-update-graph')
])

# EN: Callback to update graph dynamically when slider changes
# ES: Callback para actualizar el gráfico dinámicamente cuando cambia el slider
@app.callback(
    Output('live-update-graph', 'figure'),
    Input('slider', 'value')
)
def update_graph(idx):
    # EN: Select X and Y features based on slider position
    # ES: Seleccionar características X y Y según la posición del slider
    x_feat = num_cols[idx]
    # wrap around to avoid out-of-range
    y_feat = num_cols[(idx + 1) % len(num_cols)]
    # EN: Create scatter plot with histogram marginals
    # ES: Crear gráfico de dispersión con histogramas marginales
    fig = px.scatter(
        df,
        x=x_feat,
        y=y_feat,
        color='species',
        title=f'{x_feat} vs {y_feat}',
        marginal_x='histogram',
        marginal_y='histogram'
    )
    return fig

# EN: Run Dash application
# ES: Ejecutar la aplicación Dash
if __name__ == '__main__':
    app.run(debug=True, port=8050)
