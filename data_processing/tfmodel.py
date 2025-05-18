import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# === CONFIGURACIÓN ===
ruta_jsonl = "landmarks_con_errores.jsonl"
landmarks_usados = [
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_HIP", "RIGHT_HIP"
]
errores_keys = [
    "pies_a_la_anchura_de_hombros",
    "agarre_amplio",
    "espalda_neutral",
    "hombros_sobre_barra"
]

# === CARGAR Y PREPARAR DATOS ===
X = []
y = []

with open(ruta_jsonl, "r") as f:
    for index, line in enumerate(f):
        data = json.loads(line)
        puntos = data.get("landmarks", {})
        errores = data.get("errores", {})

        # Vector de entrada (30 valores = 10 puntos x 3 coords)
        vector = []
        skip = False
        for nombre in landmarks_usados:
            punto = puntos.get(nombre)
            if punto is None:
                vector.extend([-1.0, -1.0, -1.0])  # valor de relleno para landmark faltante
            else:
                vector.extend(punto)
        X.append(vector)


        # Vector de salida multietiqueta
        y_vector = []
        for err in errores_keys:
            val = errores.get(err)
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
    Dense(4, activation='sigmoid')  # 4 errores posibles
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
model.save("modelo_multietiqueta_deadlift.keras")
print("✅ Modelo guardado como modelo_multietiqueta_deadlift.keras")
