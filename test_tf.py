import tensorflow as tf

print("¡TensorFlow está funcionando!")
print("Versión instalada:", tf.__version__)

# Una pequeña prueba matemática
print("Prueba de cálculo:", tf.add(10, 20).numpy())