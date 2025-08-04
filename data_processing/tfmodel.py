import time
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# === CONFIGURACIÓN ===
ruta_jsonl = "all_vectors_v3.jsonl"

landmarks_usados = [
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_HIP", "RIGHT_HIP"
]
valores = [
    "pies_a_la_anchura_de_hombros",
    "agarre_amplio",
    "espalda_neutral",
    "hombros_sobre_barra",
]

# === CARGAR Y PREPARAR DATOS ===
X = []
y = []

with open(ruta_jsonl, "r") as f:
    for index, line in enumerate(f):
        data = json.loads(line)
        puntos = data.get("landmarks", {})
        resultados = data.get("resultados", {})

        # Vector de entrada (30 valores = 10 puntos x 3 coords)
        vector = []
        
        for nombre in landmarks_usados:
            punto = puntos.get(nombre)
            vector.extend(punto)
        X.append(vector)

        # Vector de salida multietiqueta
        y_vector = []
        for dato in valores:
            val = resultados.get(dato)
            y_vector.append(0 if val is True else 1)  # 1 si hay error o es None
        y.append(y_vector)
        

print(index)
X = np.array(X)
y = np.array(y)

print("Ejemplos cargados:", len(X))
print("Shape de entrada:", X.shape)
print("Shape de salida:", y.shape)

# === ENTRENAR MODELO MULTIETIQUETA ===
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(4, activation='sigmoid')  # 5 salidas para 5 etiquetas
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

# === GUARDAR MODELO ENTRENADO ===
model.save("Model_v3.keras")
print("✅ Modelo guardado")
