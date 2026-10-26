# 📊 Pronóstico de Series Temporales con Deep Learning y Streamlit

Bienvenido al repositorio de **Analítica y Pronóstico con TensorFlow y Deep Learning**. Este proyecto implementa modelos de inteligencia artificial para la predicción de series temporales, destacando un **Dashboard Interactivo en Streamlit** impulsado por una red neuronal recurrente **LSTM (Long Short-Term Memory) Multivariada** para predecir la demanda y venta de tiquetes de transporte por ruta y franja horaria.

---

## 🚀 Tabla de Contenidos

- [Descripción General](#-descripción-general)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Requisitos Previos](#-requisitos-previos)
- [Guía Paso a Paso: Instalación y Configuración](#-guía-paso-a-paso-instalación-y-configuración)
  - [Paso 1: Clonar o abrir el repositorio](#paso-1-clonar-o-abrir-el-repositorio)
  - [Paso 2: Crear el entorno virtual](#paso-2-crear-el-entorno-virtual)
  - [Paso 3: Activar el entorno virtual](#paso-3-activar-el-entorno-virtual)
  - [Paso 4: Instalar dependencias](#paso-4-instalar-dependencias)
  - [Paso 5: Validar la instalación de TensorFlow y Keras](#paso-5-validar-la-instalación-de-tensorflow-y-keras)
- [Ejecución del Proyecto](#-ejecución-del-proyecto)
  - [1. Dashboard Interactivo de Streamlit](#1-dashboard-interactivo-de-streamlit)
  - [2. Prueba Rápida con Red Neuronal Básica](#2-prueba-rápida-con-red-neuronal-básica)
  - [3. Cuadernos de Jupyter (Análisis Exploratorio)](#3-cuadernos-de-jupyter-análisis-exploratorio)
- [Arquitectura del Modelo LSTM Multivariado](#-arquitectura-del-modelo-lstm-multivariado)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Solución de Problemas Frecuentes (FAQ)](#-solución-de-problemas-frecuentes-faq)

---

## 📖 Descripción General

El núcleo del proyecto es resolver problemas de estimación de demanda en el sector transporte mediante Deep Learning:
- **Modelo LSTM Multivariado:** Captura dependencias temporales complejas considerando no solo el historial de ventas, sino variables exógenas como ruta de viaje, franja horaria (Mañana, Tarde, Noche), día de la semana y mes del año.
- **Visualización Interactiva:** Panel de control web desarrollado con **Streamlit** y gráficos dinámicos con **Plotly**, permitiendo a tomadores de decisiones filtrar rutas, comparar datos reales vs. pronósticos en tiempo real y visualizar KPIs clave.
- **Laboratorio de Aprendizaje:** Incluye scripts introductorios a redes neuronales densas y notebooks experimentales con análisis univariado y multivariado.

---

## 📂 Estructura del Repositorio

```plaintext
tensorflow/
├── dashboard_streamlit.py   # Aplicación web interactiva (Streamlit + LSTM + Plotly)
├── red_neuronal.py          # Script de iniciación a redes neuronales (regresión y = 2x - 1)
├── test_tf.py               # Script de verificación para TensorFlow
├── test_keras.py            # Script de verificación para Keras
├── LSTM.ipynb               # Notebook con desarrollo y experimentación del modelo LSTM
├── ejemplo.ipynb            # Notebook de análisis y pronóstico de series temporales univariadas
├── Untitled.ipynb           # Cuaderno de pruebas y prototipado
├── time_series.csv          # Dataset histórico univariado (fechas y unidades vendidas)
├── pronostico.csv           # Resultados exportados de pronósticos generados
├── requirements.txt         # Lista de dependencias del proyecto
├── setup.bat                # Script de automatización para Windows
└── README.md                # Documentación del proyecto
```

---

## 🛠 Requisitos Previos

Antes de comenzar, asegúrate de tener instalado en tu sistema:
- **Python:** Versión recomendada **3.10**, **3.11** o **3.12**.
- **Git:** Para clonar y gestionar el repositorio.
- **PowerShell / Terminal de comandos** (en Windows) o bash/zsh (en Linux/macOS).

Verifica tu versión de Python ejecutando:
```bash
python --version
```

---

## 📋 Guía Paso a Paso: Instalación y Configuración

Sigue estos pasos en orden para poner en marcha el entorno:

### Paso 1: Clonar o abrir el repositorio
Si aún no has clonado el repositorio:
```bash
git clone https://github.com/01gr35/tensorflow.git
cd tensorflow
```
Si ya te encuentras en el directorio del proyecto, abre una terminal en la raíz (`d:\Datos\Analitica\tensorflow`).

---

### Paso 2: Crear el entorno virtual
Es una buena práctica aislar las dependencias del proyecto creando un entorno virtual (`.venv`):

```bash
python -m venv .venv
```

---

### Paso 3: Activar el entorno virtual

- **En Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(Si recibes un error de permisos o script execution policy, ejecuta primero: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

- **En Windows (CMD / Símbolo del sistema):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

- **En Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

> 💡 **Nota para Windows:** También puedes utilizar el archivo `setup.bat` para automatizar la activación y actualización de dependencias.

Al activarse, verás `(.venv)` al inicio de la línea de tu terminal.

---

### Paso 4: Instalar dependencias

Con el entorno virtual activado, actualiza `pip` e instala todas las librerías necesarias:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Librerías principales instaladas:
- `tensorflow`: Motor de Deep Learning y Redes Neuronales.
- `streamlit`: Servidor y framework para construir el dashboard interactivo.
- `plotly`: Visualizaciones interactivas de series de tiempo.
- `scikit-learn`: Preprocesamiento, normalización y escalado de datos (`MinMaxScaler`).
- `pandas` & `numpy`: Manipulación y transformación de estructuras de datos.
- `matplotlib`: Gráficos estáticos complementarios.

---

### Paso 5: Validar la instalación de TensorFlow y Keras

Ejecuta los scripts de prueba para certificar que el backend de TensorFlow y Keras funcionan correctamente:

```bash
# Validar TensorFlow
python test_tf.py

# Validar Keras
python test_keras.py
```

Si todo está configurado correctamente, verás en consola las versiones instaladas y un cálculo de prueba exitoso.

---

## 💻 Ejecución del Proyecto

### 1. Dashboard Interactivo de Streamlit

Para iniciar la aplicación web principal:

```bash
streamlit run dashboard_streamlit.py
```

Streamlit abrirá automáticamente una pestaña en tu navegador web predeterminado (por defecto en `http://localhost:8501`).

#### Características del Dashboard:
1. **Filtros de Negocio (Barra Lateral):**
   - Selección de Ruta (ej. *Bogota-Medellin*, *Cali-Bogota*).
   - Selección de Horario (ej. *Mañana*, *Tarde*, *Noche*).
2. **KPIs en Tiempo Real:**
   - Total de tiquetes vendidos en el periodo.
   - Promedio diario real de ventas.
   - Pronóstico promedio generado por la red LSTM.
3. **Gráfico Interactivo de Tendencia (Plotly):**
   - Comparación simultánea entre la curva de ventas reales vs. pronóstico LSTM.
   - Modo de inspección hover sincronizado (`x unified`).
4. **Tabla Detallada:**
   - Desglose cronológico de fechas con la comparativa `Dato Real` vs. `Dato Pronosticado`.

---

### 2. Prueba Rápida con Red Neuronal Básica

Para ver cómo entrena una red neuronal artificial simple de una sola neurona para aprender una función lineal ($y = 2x - 1$):

```bash
python red_neuronal.py
```

---

### 3. Cuadernos de Jupyter (Análisis Exploratorio)

Si deseas inspeccionar el desarrollo paso a paso del modelo LSTM o las pruebas univariadas:

```bash
# Iniciar Jupyter
jupyter notebook
```
- **`LSTM.ipynb`**: Contiene la generación de los datos sintéticos multivariados, la construcción de tensores en 3D para secuencias temporales, el entrenamiento de la capa LSTM y la inversión de escala.
- **`ejemplo.ipynb`**: Experimentación clásica sobre series temporales del archivo `time_series.csv` y generación del archivo `pronostico.csv`.

---

## 🧠 Arquitectura del Modelo LSTM Multivariado

El flujo analítico implementado en `dashboard_streamlit.py` opera de la siguiente manera:

1. **Ingeniería de Características:**
   - Variables categóricas codificadas mediante *One-Hot Encoding* (`Ruta`, `Horario`, `Dia_Semana`, `Mes`).
   - Normalización de todas las características en el rango $[0, 1]$ mediante `MinMaxScaler`.
2. **Generación de Ventanas Deslizantes (*Sliding Windows*):**
   - Se utiliza un tamaño de ventana temporal de **14 días** (`ventana = 14`), transformando los datos en un tensor tridimensional con forma `[muestras, pasos_de_tiempo, características]`.
3. **Estructura de la Red:**
   - **Capa LSTM:** 32 unidades con función de activación predeterminada para capturar dependencias temporales.
   - **Dropout (10%):** Previene el sobreajuste (*overfitting*).
   - **Capa Densa de Salida:** 1 neurona lineal encargada de predecir las ventas del siguiente periodo.
4. **Función de Pérdida y Optimizador:**
   - Optimizador: `adam`.
   - Función de pérdida: Error Cuadrático Medio (`mean_squared_error`).

---

## 🧰 Tecnologías Utilizadas

| Componente | Tecnología | Propósito |
| :--- | :--- | :--- |
| **Lenguaje** | Python 3.12 | Lenguaje de programación principal |
| **Deep Learning** | TensorFlow / Keras | Construcción y entrenamiento de la red LSTM |
| **Interfaz Web** | Streamlit | Creación del Dashboard analítico interactivo |
| **Visualización** | Plotly & Matplotlib | Gráficos interactivos y dinámicos |
| **Data Science** | Pandas & NumPy | Procesamiento de datasets y cálculos matriciales |
| **Machine Learning** | Scikit-Learn | Escalado y normalización de variables |

---

## ❓ Solución de Problemas Frecuentes (FAQ)

### 1. Error de ejecución de scripts en PowerShell (`Activate.ps1 cannot be loaded`)
PowerShell restringe la ejecución de scripts por seguridad. Puedes permitirlo para la sesión actual ejecutando:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Error `ModuleNotFoundError: No module named 'streamlit'`
Asegúrate de haber activado el entorno virtual (`.venv`) y haber ejecutado:
```bash
pip install -r requirements.txt
```

### 3. El dashboard tarda unos segundos en iniciar la primera vez
Es el comportamiento normal: durante la primera ejecución, la función decorada con `@st.cache_data` entrena la red neuronal LSTM y genera las predicciones. En subsecuentes interacciones, Streamlit utiliza la memoria caché para ofrecer una navegación instantánea.

---

## 👥 Autor y Créditos

Desarrollado como proyecto de analítica avanzada y predicción de series de tiempo con Inteligencia Artificial.
Repositorio oficial: [01gr35/tensorflow](https://github.com/01gr35/tensorflow)

