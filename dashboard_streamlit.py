import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go # <-- IMPORTANTE: Importamos Plotly
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Configuración de la página del Dashboard
st.set_page_config(page_title="Dashboard de Pronóstico de Tiquetes", layout="wide")

st.title("📊 Dashboard Inteligente: Pronóstico de Ventas de Tiquetes")
st.markdown("Este panel utiliza una red neuronal **LSTM Multivariada** para predecir la demanda por ruta y franja horaria.")

# ====================================================================
# 1. CARGA Y SIMULACIÓN DE DATOS (Se usa caché para no regenerar siempre)
# ====================================================================
@st.cache_data
def cargar_datos_y_entrenar():
    np.random.seed(42)
    rutas = ['Bogota-Medellin', 'Cali-Bogota']
    horarios = ['Mañana', 'Tarde', 'Noche']
    fechas = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")

    lista_datos = []
    for fecha in fechas:
        for ruta in rutas:
            for horario in horarios:
                base = 40 if ruta == 'Bogota-Medellin' else 25
                modificador_horario = 15 if horario == 'Mañana' else (5 if horario == 'Tarde' else 20)
                fin_de_semana = 30 if fecha.weekday() in [4, 5, 6] else 0
                ruido = np.random.normal(0, 5)
                ventas = max(0, int(base + modificador_horario + fin_de_semana + ruido))
                
                lista_datos.append({
                    'Fecha': fecha, 'Ruta': ruta, 'Horario': horario,
                    'Dia_Semana': fecha.weekday(), 'Mes': fecha.month, 'Ventas': ventas
                })

    df_base = pd.DataFrame(lista_datos)
    
    # Preprocesamiento para la IA
    df_procesado = pd.get_dummies(df_base, columns=['Ruta', 'Horario', 'Dia_Semana', 'Mes'], dtype=int)
    columnas = [col for col in df_procesado.columns if col not in ['Fecha', 'Ventas']] + ['Ventas']
    df_final = df_procesado[columnas]

    # Escalado
    scaler = MinMaxScaler(feature_range=(0, 1))
    datos_escalados = scaler.fit_transform(df_final)

    # Ventanas temporales
    ventana = 14
    X, y = [], []
    for i in range(len(datos_escalados) - ventana):
        X.append(datos_escalados[i:(i + ventana), :])
        y.append(datos_escalados[i + ventana, -1])
    X, y = np.array(X), np.array(y)

    # Modelo y Entrenamiento rápido
    model = Sequential([
        LSTM(units=32, return_sequences=False, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.1),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=128, verbose=0) # Pocas épocas para agilizar el dashboard

    # Predicciones globales
    predicciones = model.predict(X)
    matriz_espejo = np.zeros((len(predicciones), datos_escalados.shape[1]))
    matriz_espejo[:, -1] = predicciones.flatten()
    predicciones_reales = scaler.inverse_transform(matriz_espejo)[:, -1]

    # Alinear predicciones al DataFrame original (perdemos los primeros 'ventana' elementos)
    df_resultados = df_base.iloc[ventana:].copy()
    df_resultados['Prediccion_LSTM'] = predicciones_reales
    
    return df_resultados

# Ejecutar proceso analítico
with st.spinner("Entrenando red neuronal y preparando analíticas..."):
    df_dashboard = cargar_datos_y_entrenar()

# ====================================================================
# 2. CONTROLES DE LA BARRA LATERAL (SIDEBAR)
# ====================================================================
st.sidebar.header("Filtros de Negocio")
ruta_seleccionada = st.sidebar.selectbox("Selecciona la Ruta:", df_dashboard['Ruta'].unique())
horario_seleccionado = st.sidebar.selectbox("Selecciona el Horario:", df_dashboard['Horario'].unique())

# Filtrado dinámico del DataFrame según la selección del usuario
df_filtrado = df_dashboard[(df_dashboard['Ruta'] == ruta_seleccionada) & 
                            (df_dashboard['Horario'] == horario_seleccionado)].tail(90) # Últimos 90 días

# ====================================================================
# 3. MÉTRICAS CLAVE (KPIs)
# ====================================================================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Tiquetes Vendidos (Periodo)", f"{int(df_filtrado['Ventas'].sum()):,}")
with col2:
    st.metric("Promedio Diario Real", f"{int(df_filtrado['Ventas'].mean())} uds")
with col3:
    st.metric("Pronóstico Promedio LSTM", f"{int(df_filtrado['Prediccion_LSTM'].mean())} uds")

st.markdown("---")

# ====================================================================
# 4. GRÁFICA INTERACTIVA AJUSTADA (Ahora con Plotly)
# ====================================================================
st.subheader(f"Tendencia y Predicción para: {ruta_seleccionada} ({horario_seleccionado})")

# Crear la figura de Plotly
fig = go.Figure()

# Agregar la línea de Ventas Reales
fig.add_trace(go.Scatter(
    x=df_filtrado['Fecha'], 
    y=df_filtrado['Ventas'], 
    mode='lines', 
    name='Ventas Reales',
    line=dict(color='teal', width=2),
    hovertemplate='<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Ventas Reales:</b> %{y} uds<extra></extra>'
))

# Agregar la línea de Pronóstico
fig.add_trace(go.Scatter(
    x=df_filtrado['Fecha'], 
    y=df_filtrado['Prediccion_LSTM'], 
    mode='lines', 
    name='Pronóstico LSTM',
    line=dict(color='crimson', width=2, dash='dash'),
    hovertemplate='<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Pronóstico:</b> %{y:.0f} uds<extra></extra>'
))

# Configurar el diseño del gráfico
fig.update_layout(
    yaxis_title='Cantidad de Tiquetes',
    hovermode='x unified', # Muestra ambos valores al mismo tiempo al pasar el cursor
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=40, t=40, b=40)
)

# Renderizar en Streamlit (usar_container_width hace que ocupe todo el ancho)
st.plotly_chart(fig, use_container_width=True)

# ====================================================================
# 5. NUEVA TABLA DETALLADA (Datos Reales vs Pronóstico)
# ====================================================================
st.markdown("---")
st.subheader("📋 Detalle de Datos: Real vs Pronosticado")

# Creamos una copia del dataframe filtrado para no alterar el original
df_tabla = df_filtrado.copy()

# Redondeamos los pronósticos a números enteros para que se vean mejor
df_tabla['Prediccion_LSTM'] = df_tabla['Prediccion_LSTM'].round().astype(int)

# Seleccionamos y renombramos las columnas según tu requerimiento
df_tabla_final = df_tabla[['Fecha', 'Ruta', 'Horario', 'Ventas', 'Prediccion_LSTM']].copy()
df_tabla_final.columns = ['Fecha', 'Ruta', 'Hora', 'Dato Real', 'Dato Pronosticado']

# Formateamos la columna Fecha para que no muestre la hora (00:00:00)
df_tabla_final['Fecha'] = df_tabla_final['Fecha'].dt.strftime('%Y-%m-%d')

# Mostramos la tabla en Streamlit
st.dataframe(df_tabla_final, use_container_width=True, hide_index=True)