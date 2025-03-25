from flask import Flask, send_file
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for image generation
import matplotlib.pyplot as plt
import io
import pandas as pd
import numpy as np
import seaborn as sns
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

data = pd.read_csv('/Users/juansalazar/Documents/Datathon_2025_Interlub/utils/Datathon.csv', parse_dates=['Creacion Orden de Venta'])
data['Mes'] = data['Creacion Orden de Venta'].dt.to_period('M')

df_agg = data.groupby(['Articulo', 'Mes']).agg(
    Cantidad_Total=('Cantidad', 'sum'),
).reset_index()

unidad_mas_frecuente = (
    data.groupby(['Articulo', 'Mes'])['Unidad de venta']
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index()
)

df_final = pd.merge(df_agg, unidad_mas_frecuente, on=['Articulo', 'Mes'])

# Endpoint 1: Consumo por artículo y mes
@app.route('/generate-image')
def generate_image():
    articulo = 'IVP01004'
    df = df_final[df_final['Articulo'] == articulo].copy()
    df['Mes'] = df['Mes'].astype('str')
    plt.figure(figsize=(8, 5))

    if len(df) > 1:
        sns.histplot(df, x='Mes', weights='Cantidad_Total', bins=5, color='skyblue', kde=False)
    else:
        plt.bar(df['Mes'].astype(str), df['Cantidad_Total'], color='skyblue')

    plt.title(f'Consumo del artículo {articulo} por Mes')
    plt.xlabel('Mes')
    plt.ylabel('Cantidad de Consumo')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')

# Endpoint 2: Consumo por segmento
@app.route('/segmento-consumo')
def segmento_consumo():
    plt.figure(figsize=(10, 6))
    df_segmentos = data.groupby(['Segmento Cliente', 'Unidad de venta']).agg(Cantidad_Total=('Cantidad', 'sum')).reset_index()
    df_segmentos['Unidad'] = df_segmentos['Unidad de venta'].replace({'L': 'Masa/Volumen', 'KG': 'Masa/Volumen'})
    df_segmentos = df_segmentos.groupby(['Segmento Cliente', 'Unidad'], as_index=False)['Cantidad_Total'].sum()
    ax = sns.barplot(data=df_segmentos, x='Segmento Cliente', y='Cantidad_Total', hue='Unidad', palette='Set2')
    for container in ax.containers:
        ax.set_ylim(0, df_segmentos['Cantidad_Total'].max() * 1.2)
        ax.bar_label(container, fmt='%.0f', label_type='edge', fontsize=9, color='black', rotation=90, )
    plt.title('Cantidad de consumo por Segmento')
    plt.xlabel('Segmento')
    plt.ylabel('Cantidad')
    plt.xticks(rotation=90)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')

# Endpoint 3: Top 25 artículos más consumidos
@app.route('/top-articulos')
def top_articulos():
    df_top_articulos = data.groupby(['Articulo', 'Unidad de venta']).agg(Cantidad=('Cantidad', 'sum')).reset_index()
    df_top_articulos['Unidad'] = df_top_articulos['Unidad de venta'].replace({'L': 'Masa/Volumen', 'KG': 'Masa/Volumen'})
    df_top_articulos = df_top_articulos.groupby(['Articulo', 'Unidad'], as_index=False)['Cantidad'].sum()
    df_articulos_top25 = df_top_articulos.groupby(['Articulo']).agg(Cantidad=('Cantidad', 'sum')).reset_index()
    articulos_top25 = df_articulos_top25.sort_values(by='Cantidad', ascending=False).head(25)['Articulo']
    df_top_articulos = df_top_articulos[df_top_articulos['Articulo'].isin(articulos_top25)].sort_values(by='Cantidad', ascending=False)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_top_articulos, x='Articulo', y='Cantidad', hue='Unidad', palette='Set2')
    for container in ax.containers:
        ax.set_ylim(0, df_top_articulos['Cantidad'].max() * 1.2)
        ax.bar_label(container, fmt='%.0f', label_type='edge', fontsize=9, color='black', rotation=90, padding=5)
    plt.title('Top 25 artículos más consumidos')
    plt.xlabel('Artículo')
    plt.ylabel('Cantidad')
    plt.xticks(rotation=90)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')

# Endpoint 4: Top 25 clientes que consumen más
@app.route('/top-clientes')
def top_clientes():
    df_top_clientes = data.groupby(['Codigo Cliente', 'Unidad de venta']).agg(Cantidad=('Cantidad', 'sum')).reset_index()
    df_top_clientes['Unidad'] = df_top_clientes['Unidad de venta'].replace({'L': 'Masa/Volumen', 'KG': 'Masa/Volumen'})
    df_top_clientes = df_top_clientes.groupby(['Codigo Cliente', 'Unidad'], as_index=False)['Cantidad'].sum()
    df_clientes_top25 = df_top_clientes.groupby(['Codigo Cliente']).agg(Cantidad=('Cantidad', 'sum')).reset_index()
    clientes_top = df_clientes_top25.sort_values(by='Cantidad', ascending=False).head(25)['Codigo Cliente']
    df_top_clientes = df_top_clientes[df_top_clientes['Codigo Cliente'].isin(clientes_top)].sort_values(by='Cantidad', ascending=False)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_top_clientes, x='Codigo Cliente', y='Cantidad', hue='Unidad', palette='Set2')
    for container in ax.containers:
        ax.set_ylim(0, df_top_clientes['Cantidad'].max() * 1.2)
        ax.bar_label(container, fmt='%.0f', label_type='edge', fontsize=9, color='black', rotation=90, padding=5)
    plt.title('Top 25 clientes que consumen más')
    plt.xlabel('Codigo Cliente')
    plt.ylabel('Cantidad')
    plt.xticks(rotation=90)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)