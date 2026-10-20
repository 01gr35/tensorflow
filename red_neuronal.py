import tensorflow as tf
import numpy as np

print("1. Creando los datos de entrenamiento...")
# X = Entradas (del -1 al 1)
x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
# Y = Salidas esperadas (la fórmula es y = 2x - 1)
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

print("2. Construyendo la red neuronal...")
# Creamos un modelo secuencial con una sola capa (Dense) y una neurona
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

print("3. Compilando el modelo...")
# Optimizador 'adam' y función de pérdida para medir el error
modelo.compile(optimizer='adam', loss='mean_squared_error')

print("4. Entrenando a la IA (esto puede tardar unos segundos)...")
# Entrenamos el modelo 500 veces con nuestros datos
modelo.fit(x, y, epochs=500, verbose=0)

print("5. ¡Entrenamiento terminado! Vamos a probar:")
# Le pedimos que prediga el valor para X = 10
# Esperamos que el resultado sea cercano a 19 (2*10 - 1)
prediccion = modelo.predict(np.array([10.0]))
print(f"Si X es 10, la IA predice que Y es: {prediccion[0][0]:.2f}")