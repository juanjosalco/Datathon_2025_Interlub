from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import seaborn as sns

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from scipy.stats import kruskal
import math
import pymannkendall as mk
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tbats import TBATS
from prophet import Prophet
from pmdarima import auto_arima


import matplotlib
matplotlib.use('Agg')  # Use Agg backend for image generation
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)


def linear_regresion(filtered_df):

    filtered_df['Year'] = filtered_df['Week'].str.extract(r'(\d{4})').astype(int)
    filtered_df['Week_Num'] = filtered_df['Week'].str.extract(r'Week(\d+)').astype(int)
    filtered_df['Week_Num'] += np.where(filtered_df['Year'] == 2022, 52, 0)
    filtered_df['Week_Num'] += np.where(filtered_df['Year'] == 2023, 104, 0)

    filtered_df = filtered_df.sort_values(by=['Articulo', 'Week_Num'])

    filtered_df['Cantidad_Acumulada'] = filtered_df.groupby('Articulo')['Cantidad'].cumsum()

   
    X = filtered_df['Week_Num'].values.reshape(-1, 1)
    y = filtered_df['Cantidad_Acumulada'].values

    X_train, X_test = X[:-5], X[-5:]
    y_train, y_test = y[:-5], y[-5:]

    model = LinearRegression()
    model.fit(X, y)
    score = model.score(X_test, y_test)

    return model , X, y, score

def promedios_movil(df, articulo):

    df_resultado, df_diferencias, df_predicciones, unidad_venta, pred_weeks, promedio_cantidad, ultimas_semanas = modelos_series_tiempo(df, articulo)

    return df_resultado, df_diferencias, df_predicciones, unidad_venta, pred_weeks, promedio_cantidad, ultimas_semanas

def filtrar_por_articulo(df, codigo_articulo):
    filter_df = df[df['Articulo'] == codigo_articulo]
    return filter_df

def filtrar_por_cantidad_de_filas(df, min_filas=30, max_filas=79):
    articulo_counts = df['Articulo'].value_counts()
    valid_articulos = articulo_counts[(articulo_counts > min_filas) & (articulo_counts < max_filas)].index
    return df[df['Articulo'].isin(valid_articulos)]

def calcular_diferencias_semanas(df, articulo):
    semanas = df[df["Articulo"] == articulo]["Week Number"].sort_values().unique()
    diferencias = np.diff(semanas)  # Calcula la diferencia entre semanas consecutivas
    return diferencias

def modelos_series_tiempo(df, articulo, window=3):
    df_articulo = df[df["Articulo"] == articulo].copy()

    # --- Modelado para Cantidad ---
    # Promedio Móvil
    df_articulo["PromedioMovil"] = df_articulo["Cantidad"].rolling(window=window).mean()

    # Promedio Móvil Ponderado
    pesos = np.arange(1, window + 1)
    df_articulo["PromedioMovilPonderado"] = df_articulo["Cantidad"].rolling(window).apply(lambda x: np.dot(x, pesos) / pesos.sum(), raw=True)

    # Suavización Exponencial
    modelo_exp = SimpleExpSmoothing(df_articulo["Cantidad"].dropna()).fit(optimized=True)
    df_articulo["SuavExp"] = modelo_exp.fittedvalues

    # --- Modelado para Diferencias de Semanas ---
    diferencias = calcular_diferencias_semanas(df, articulo)
    if len(diferencias) >= window:
        df_diferencias = pd.DataFrame({"Diff Weeks": diferencias})

        # Promedio Móvil
        df_diferencias["PromedioMovil"] = df_diferencias["Diff Weeks"].rolling(window=window).mean()

        # Promedio Móvil Ponderado
        df_diferencias["PromedioMovilPonderado"] = df_diferencias["Diff Weeks"].rolling(window).apply(lambda x: np.dot(x, pesos) / pesos.sum(), raw=True)

        # Suavización Exponencial
        modelo_exp_diff = SimpleExpSmoothing(df_diferencias["Diff Weeks"].dropna()).fit(optimized=True)
        df_diferencias["SuavExp"] = modelo_exp_diff.fittedvalues

        # Predicciones de Cantidad
        pred_cantidad_movil = df_articulo["PromedioMovil"].iloc[-1]
        pred_cantidad_movil_pond = df_articulo["PromedioMovilPonderado"].iloc[-1]
        pred_cantidad_suav_exp = df_articulo["SuavExp"].iloc[-1]
        promedio_cantidad = (pred_cantidad_movil + pred_cantidad_movil_pond + pred_cantidad_suav_exp) / 3

        # Predicciones de Semanas
        pred_semana_movil = df_diferencias["PromedioMovil"].iloc[-1]
        pred_semana_movil_pond = df_diferencias["PromedioMovilPonderado"].iloc[-1]
        pred_semana_suav_exp = df_diferencias["SuavExp"].iloc[-1]
        promedio_diferencias = (pred_semana_movil + pred_semana_movil_pond + pred_semana_suav_exp) / 3

        ultimas_semanas = df_articulo["Week Number"].max()
        unidad_venta = df_articulo["Unidad de venta"].iloc[0]

        # Calcular la semana de la próxima venta
        pred_weeks = math.ceil(promedio_diferencias)

    df_predicciones = pd.DataFrame({
        "Week Number": [pred_weeks],  # Convertir a lista para evitar errores
        "Última semana de venta": [ultimas_semanas],
        "Cantidad": [promedio_cantidad],
        "Unidad": [unidad_venta]
    })

    return df_articulo, df_diferencias, df_predicciones, unidad_venta, pred_weeks, promedio_cantidad, ultimas_semanas

def completar_semanas_para_articulo(df, articulo_interes):
    df["Week"] = df["Week"].astype(str)
    df_articulo = df[df["Articulo"] == articulo_interes].copy()
    df_articulo["Year"] = df_articulo["Week"].str[:4]
    años = df_articulo["Year"].unique()
    semanas = [f"{year}-{str(semana).zfill(2)}" for year in años for semana in range(1, 53)]
    df_completo = pd.DataFrame({"Week": semanas})
    df_completo["Articulo"] = articulo_interes
    df_completo["Week"] = df_completo["Week"].astype(str)
    df_merged = df_completo.merge(df_articulo, on=['Week', 'Articulo'], how='left')
    df_merged["Cantidad"] = df_merged["Cantidad"].fillna(0)

    return df_merged

def seleccionar_modelo2(df, articulo, semanas_pred):
    
    df_articulo = df[df["Articulo"] == articulo].copy()
    df_articulo = df_articulo.sort_values(by="Week2")

    # Extraer número de semana para prueba de estacionalidad
    df_articulo["Week_Number"] = df_articulo["Week2"].dt.isocalendar().week
    ventas_por_semana = [df_articulo[df_articulo["Week_Number"] == w]["Cantidad"].values for w in df_articulo["Week_Number"].unique()]

    ventas_por_semana = [df_articulo[df_articulo["Week_Number"] == w]["Cantidad"].values 
                     for w in df_articulo["Week_Number"].unique()]
    
    # Prueba de Kruskal-Wallis
    # estadistico, p_value = kruskal(*ventas_por_semana)
    # estacional = p_value < 0.05
    # print(f"Valor p Kruskal-Wallis: {p_value:.5f}")
    # print(f"{'Existe' if estacional else 'NO Existe'} evidencia de estacionalidad en {articulo}")

    estacional = False
    # Preparar datos
    y = df_articulo["Cantidad"].values
    fechas = df_articulo["Week2"]

    # Horizonte de predicción
    futuras_fechas = pd.date_range(start=fechas.max(), periods=semanas_pred+1, freq="W")[1:]

    predicciones = {}

    if not estacional:
        # Modelos SIN estacionalidad
        ses_model = SimpleExpSmoothing(y).fit()
        predicciones["SES"] = ses_model.forecast(semanas_pred)

        holt_model = ExponentialSmoothing(y, trend="add").fit()
        predicciones["Holt"] = holt_model.forecast(semanas_pred)

        arima_model = auto_arima(y, seasonal=False, stepwise=True, suppress_warnings=True)
        predicciones["ARIMA"] = arima_model.predict(n_periods=semanas_pred)
    else:
        # Modelos CON estacionalidad
        sarima_model = auto_arima(y, seasonal=True, m=52, stepwise=True, suppress_warnings=True)
        predicciones["SARIMA"] = sarima_model.predict(n_periods=semanas_pred)

        ets_model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=52).fit()
        predicciones["ETS"] = ets_model.forecast(semanas_pred)

        df_prophet = pd.DataFrame({"ds": fechas, "y": y})
        prophet_model = Prophet()
        prophet_model.fit(df_prophet)
        futuro = pd.DataFrame({"ds": futuras_fechas})
        predicciones["Prophet"] = prophet_model.predict(futuro)["yhat"].values

    # Crear DataFrame con predicciones
    
    df_predicciones = pd.DataFrame({"Week2": futuras_fechas, "SARIMA": None, "ETS": None, "Prophet": None})

    for modelo, pred in predicciones.items():
        df_predicciones[modelo] = pred

    # Calcular predicción promedio
    df_predicciones["Promedio"] = df_predicciones[["SARIMA", "ETS", "Prophet"]].mean(axis=1)

    predicciones_res= []

    for index, row in df.iterrows():
        predicciones_res.append(row)

    promedios = []
    # Imprimir predicciones
    for i, row in df_predicciones.iterrows():
        promedios.append(row['Promedio'])
        
    return df_predicciones, fechas, predicciones, futuras_fechas, estacional, promedios,y

@app.route('/predictions', methods=['POST'])
def predictions():

    data = request.get_json()

    article = data["id"]
    prediction_time = data["time"]

    df = pd.read_excel("/Users/juansalazar/Documents/Datathon_2025_Interlub/utils/FinalWeeklyAggregatedData_F.xls", parse_dates=['Creacion Orden de Venta'])
    
    filtered_df = filtrar_por_articulo(df, article)  
    articulo_counts = filtered_df['Articulo'].value_counts().get(article, 0)
    print("!!!!!!!!!!!!!!!!")
    print(articulo_counts)
   

    # datos insuficientes
    if articulo_counts < 5:
        response = {"weeks": [0]}

    # promedios moviles
    elif articulo_counts <= 35 :
        df = pd.read_csv("weekly_aggregated_data-8(1).csv")
        print("here2")
        filtered_df = filtrar_por_articulo(df, article)  

        df_resultado, df_diferencias, df_predicciones, unidad_venta, pred_weeks, promedio_cantidad, ultimas_semanas  = promedios_movil(filtered_df, article)

        plt.figure(figsize=(12, 6))

        df_completo = df_resultado.set_index("Week Number").reindex(range(df_resultado["Week Number"].min(), df_resultado["Week Number"].max() + 1), fill_value=0).reset_index()

        # Filtrar solo semanas con ventas mayores a 0 para el eje X y reducir etiquetas
        semanas_con_ventas = df_completo[df_completo["Cantidad"] > 0]["Week Number"]
        semanas_mostradas = semanas_con_ventas[::max(1, len(semanas_con_ventas) // 10)]

        # Graficar ventas reales
        plt.plot(df_completo["Week Number"], df_completo["Cantidad"], marker='o', linestyle='-', markersize=4, label='Ventas reales', color='blue')

        # Graficar modelos con líneas más suaves
        plt.plot(df_completo["Week Number"], df_completo["PromedioMovil"], linestyle='dashed', color='orange', alpha=0.7, linewidth=1.5, label='Promedio Móvil')
        plt.plot(df_completo["Week Number"], df_completo["PromedioMovilPonderado"], linestyle='dashed', color='green', alpha=0.7, linewidth=1.5, label='Promedio Móvil Ponderado')
        plt.plot(df_completo["Week Number"], df_completo["SuavExp"], linestyle='dashed', color='red', alpha=0.7, linewidth=1.5, label='Suavización Exponencial')

        # Agregar punto de predicción
        plt.scatter([pred_weeks+ultimas_semanas], df_predicciones["Cantidad"], color='red', s=100, label='Predicción', edgecolors='black', zorder=3)

        # Mejorar formato del eje X
        plt.xticks(semanas_mostradas, rotation=45, fontsize=10)
        plt.xlabel("Semana", fontsize=12)
        plt.ylabel(f"Cantidad Vendida en {unidad_venta}", fontsize=12)
        plt.title(f"Predicción de ventas para el producto {article}", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(False)  # Desactivar cuadrícula
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        
        response = {
        'prediction' : promedio_cantidad,
        'unidad_venta' : unidad_venta
        }

    elif articulo_counts < 74:
    
        model, X, y, score = linear_regresion(filtered_df)

        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, label=f"{article} data")
        plt.plot(X, model.predict(X), linestyle='--', label=f"{article} trend")
        plt.xlabel("Semana")
        plt.ylabel("Cantidad Acumulada")
        plt.title("Regresión Lineal de Ventas Acumuladas")
        plt.legend()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        x_new =  (X[-1][0] + 1).reshape(-1, 1)
        y_pred = model.predict(x_new)[0]

        response = {
        'prediction' : y_pred
        }
        print(response)

    else:
        df = pd.read_csv("/Users/juansalazar/Documents/Datathon_2025_Interlub/utils/weekly_aggregated_data-8(1).csv")
        conteo_articulos = df['Articulo'].value_counts()
        df_mayor80= df[df['Articulo'].isin(conteo_articulos[conteo_articulos >= 80].index)]

        df_completo = completar_semanas_para_articulo(df_mayor80,article)
        df_completo['Week2'] = pd.to_datetime(df_completo['Week'].astype(str) + '-1', format='%Y-%W-%w', errors='coerce')
        df_predicciones, fechas, predicciones, futuras_fechas, estacional, promedios , y = seleccionar_modelo2(df_completo, article ,prediction_time)

        plt.figure(figsize=(12, 6))
        plt.plot(fechas, y, label="Histórico", marker="o")
        for modelo, pred in predicciones.items():
            plt.plot(futuras_fechas, pred, label=modelo)
        plt.plot(futuras_fechas, df_predicciones["Promedio"], label="Promedio", linestyle="dashed", color="black")
        plt.legend()
        plt.title(f"Pronóstico de Ventas - {article}")
        plt.xlabel("Semana")
        plt.ylabel(f"Cantidad Vendida")
        plt.grid()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        response = {
        'prediction' : promedios,
        'prueba' : estacional,
        'promedios' : promedios
        }
        

    return response

@app.route('/predictions/images', methods=['POST'])
def predictions_images():

    data = request.get_json()

    article = data["id"]
    prediction_time = data["time"]

    df = pd.read_excel("/Users/juansalazar/Documents/Datathon_2025_Interlub/utils/FinalWeeklyAggregatedData_F.xls", parse_dates=['Creacion Orden de Venta'])
    
    filtered_df = filtrar_por_articulo(df, article)  
    articulo_counts = filtered_df['Articulo'].value_counts().get(article, 0)

    # datos insuficientes
    if articulo_counts < 5:
        response = {"weeks": [0]}

    # promedios moviles
    elif articulo_counts < 30 :
        df = pd.read_csv("/Users/juansalazar/Documents/Datathon_2025_Interlub/utils/weekly_aggregated_data-8(1).csv")
        filtered_df = filtrar_por_articulo(df, article)  

        df_resultado, df_diferencias, df_predicciones, unidad_venta, pred_weeks, promedio_cantidad, ultimas_semanas  = promedios_movil(filtered_df, article)

        plt.figure(figsize=(12, 6))

        df_completo = df_resultado.set_index("Week Number").reindex(range(df_resultado["Week Number"].min(), df_resultado["Week Number"].max() + 1), fill_value=0).reset_index()

        # Filtrar solo semanas con ventas mayores a 0 para el eje X y reducir etiquetas
        semanas_con_ventas = df_completo[df_completo["Cantidad"] > 0]["Week Number"]
        semanas_mostradas = semanas_con_ventas[::max(1, len(semanas_con_ventas) // 10)]

        # Graficar ventas reales
        plt.plot(df_completo["Week Number"], df_completo["Cantidad"], marker='o', linestyle='-', markersize=4, label='Ventas reales', color='blue')

        # Graficar modelos con líneas más suaves
        plt.plot(df_completo["Week Number"], df_completo["PromedioMovil"], linestyle='dashed', color='orange', alpha=0.7, linewidth=1.5, label='Promedio Móvil')
        plt.plot(df_completo["Week Number"], df_completo["PromedioMovilPonderado"], linestyle='dashed', color='green', alpha=0.7, linewidth=1.5, label='Promedio Móvil Ponderado')
        plt.plot(df_completo["Week Number"], df_completo["SuavExp"], linestyle='dashed', color='red', alpha=0.7, linewidth=1.5, label='Suavización Exponencial')

        # Agregar punto de predicción
        plt.scatter([pred_weeks+ultimas_semanas], df_predicciones["Cantidad"], color='red', s=100, label='Predicción', edgecolors='black', zorder=3)

        # Mejorar formato del eje X
        plt.xticks(semanas_mostradas, rotation=45, fontsize=10)
        plt.xlabel("Semana", fontsize=12)
        plt.ylabel(f"Cantidad Vendida en {unidad_venta}", fontsize=12)
        plt.title(f"Predicción de ventas para el producto {article}", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(False)  # Desactivar cuadrícula
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        
        return send_file(img, mimetype='image/png')

    elif articulo_counts < 79:
    
        model, X, y, score = linear_regresion(filtered_df)

        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, label=f"{article} data")
        plt.plot(X, model.predict(X), linestyle='--', label=f"{article} trend")
        plt.xlabel("Semana")
        plt.ylabel("Cantidad Acumulada")
        plt.title("Regresión Lineal de Ventas Acumuladas")
        plt.legend()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        x_new =  (X[-1][0] + 1).reshape(-1, 1)
        y_pred = model.predict(x_new)[0]

        return send_file(img, mimetype='image/png')
    
    else:
        df_completo = completar_semanas_para_articulo(df,article)
        df_completo['Week2'] = pd.to_datetime(df_completo['Week'] + '-1', format='%Y-%W-%w')
        df_predicciones, fechas, predicciones, futuras_fechas, estacional, promedios = seleccionar_modelo2(df_completo, article ,prediction_time)

        plt.figure(figsize=(12, 6))
        plt.plot(fechas, y, label="Histórico", marker="o")
        for modelo, pred in predicciones.items():
            plt.plot(futuras_fechas, pred, label=modelo)
        plt.plot(futuras_fechas, df_predicciones["Promedio"], label="Promedio", linestyle="dashed", color="black")
        plt.legend()
        plt.title(f"Pronóstico de Ventas - {article}")
        plt.xlabel("Semana")
        plt.ylabel(f"Cantidad Vendida en -{unidad_venta}")
        plt.grid()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return send_file(img, mimetype='image/png')

@app.route('/prediction_filter', methods=['POST'])
def predictions_filter():
    print("Received POST request")

    data = request.get_json()
    if "id" not in data:
        return jsonify({"error": "Missing 'id' in request"}), 400

    article = data["id"]
    
    df = pd.read_excel("/Users/juansalazar/Documents/Datathon_2025_Interlub/utils/FinalWeeklyAggregatedData_F.xls", parse_dates=['Creacion Orden de Venta'])
    
    filtered_df = filtrar_por_articulo(df, article)  
    articulo_counts = filtered_df['Articulo'].value_counts().get(article, 0)

    print(f"Article: {article}, Count: {articulo_counts}")

    if articulo_counts < 5:
        response = {"weeks": [0]}

    elif articulo_counts < 79:
        response = {"weeks": [1]}
    else:
        response = {"weeks": [1, 2, 3]}

    return jsonify(response)



if __name__ == '__main__':
    app.run(debug=True)