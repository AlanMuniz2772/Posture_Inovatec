# Importación de librerías necesarias
import pprint                    # Para imprimir estructuras complejas de forma legible
import pandas as pd             # Para manejo de datos (aunque no se usa directamente aquí)
import tensorflow as tf         # Librería principal para redes neuronales
from data_processing.create_data_matrices import get_data  # Función para cargar datos ya procesados
import matplotlib.pyplot as plt # Para graficar
import numpy as np              # Para operaciones numéricas
from tensorflow import keras
from tensorflow.keras import layers

# Configuración: si es True, se carga un modelo ya entrenado; si es False, se entrena uno nuevo
USE_MODEL = False

# Cargar los datos de entrada (features) y salida (labels) desde un script externo
input, output = get_data()

# Separar los datos: 80% para entrenamiento y 20% para prueba
split = int(0.8 * len(input))
(train_features, train_labels), (test_features, test_labels) = (
    (input[:split], output[:split]),
    (input[split:], output[split:])
)

# Entrenamiento del modelo (solo si USE_MODEL es False)
if not USE_MODEL:

    # Definición del modelo secuencial con una sola capa densa de 5 neuronas
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5),  # Capa densa con 5 neuronas (activación lineal por defecto)
    ])

    # Definición del optimizador
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Compilación del modelo: se usa error cuadrático medio como función de pérdida
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy'],
        optimizer=opt
    )

    # Entrenamiento del modelo por 200 épocas
    hist = model.fit(train_features, train_labels, epochs=200)

    # Recuperación de métricas del historial de entrenamiento
    acc = hist.history['accuracy']
    loss = hist.history['loss']

    # Evaluación del modelo con datos de prueba
    valid_loss, valid_acc = model.evaluate(test_features, test_labels)

    # Mostrar métricas de validación
    print(f"Validation Loss: {valid_loss}\nValidation Accuracy: {valid_acc}")

    # Guardar el modelo entrenado en disco
    model.save("working_model_1")

    # Graficar la precisión durante el entrenamiento
    plt.plot(acc, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    # Crear nueva figura y graficar la pérdida durante el entrenamiento
    plt.figure()
    plt.plot(loss, label='Training Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

# Si ya existe un modelo entrenado, cargarlo y hacer predicciones
else:
    model = tf.keras.models.load_model("working_model_1")
    preds = model.predict(test_features)

    # (Este bloque no se ejecutará porque está incompleto y depende de una librería comentada)
    # cm = ConfusionMatrix(actual_vector=test_labels[0], predict_vector=preds[0])
    # print(cm.table)
