
# Modelo de Clasificación de Postura para Peso Muerto

Este modelo `modelo_multietiqueta_deadlift.keras` fue entrenado con TensorFlow/Keras para detectar errores comunes en la ejecución del ejercicio **peso muerto** usando datos de postura generados con **MediaPipe Pose**.

## 🎯 Objetivo
Detectar automáticamente los siguientes errores posturales:
1. `pies_a_la_anchura_de_hombros`
2. `agarre_amplio`
3. `espalda_neutral`
4. `hombros_sobre_barra`

---

## 🔢 Entrada esperada

- Un vector plano de **30 números** (10 landmarks × 3 coordenadas).
- Landmarks utilizados:

```
LEFT_SHOULDER, RIGHT_SHOULDER, 
LEFT_ANKLE, RIGHT_ANKLE, 
LEFT_WRIST, RIGHT_WRIST, 
LEFT_KNEE, RIGHT_KNEE, 
LEFT_HIP, RIGHT_HIP
```

- Cada landmark tiene `[x, y, z]`
- Si un punto no se detecta o tiene visibilidad baja, debe usarse `-1.0` en sus tres coordenadas.

✅ Ejemplo de entrada:

```json
[0.38, 0.25, -0.20, 0.50, 0.24, 0.12, ..., -1.0, -1.0, -1.0]
```

Funcion ejemplo: 

Función obtener_vector_para_modelo(resultados_pose, umbral = 0.5):

    Si NO hay resultados_pose.pose_landmarks:
        Devolver un vector de 30 valores con -1.0 (es decir, [−1.0, −1.0, ..., −1.0])

    Inicializar una lista vacía llamada vector_salida

    Para cada nombre en la lista de LANDMARKS_NECESARIOS (10 puntos):
        - Buscar el índice correspondiente al nombre en el enumerador de landmarks de MediaPipe
        - Obtener el punto (landmark) con ese índice de resultados_pose.pose_landmarks

        Si la visibilidad del punto es menor al umbral:
            Agregar [-1.0, -1.0, -1.0] al vector_salida
        De lo contrario:
            Agregar [punto.x, punto.y, punto.z] al vector_salida

    Devolver vector_salida como un arreglo de 30 números

---

## 🧾 Salida

- Vector de 4 probabilidades entre 0 y 1.
- Cada valor indica la probabilidad de que el error esté presente.

Ejemplo:

```json
[0.02, 0.81, 0.10, 0.67]
```

→ El modelo detecta errores en `agarre_amplio` y `hombros_sobre_barra`.

---

## 🚀 Exportar a TensorFlow.js

1. Instala el conversor:

```bash
pip install tensorflowjs
```

2. Ejecuta el comando:

```bash
tensorflowjs_converter --input_format=keras   modelo_multietiqueta_deadlift.keras   modelo_js/
```

---

## 🌐 Uso en JavaScript

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
<script>
  async function usarModelo() {
    const model = await tf.loadLayersModel('/modelo_js/model.json');
    const input = tf.tensor([[/* 30 valores */]]);
    const output = await model.predict(input).array();
    const errores = ["pies", "agarre", "espalda", "hombros"];
    output[0].forEach((v, i) => {
      if (v > 0.5) console.log("⚠️", errores[i]);
    });
  }
</script>
```

---

## 📌 Notas
- El modelo no requiere normalización extra si usas directamente la salida de MediaPipe Pose.
- Funciona offline en navegador con TensorFlow.js o puede integrarse en backend con `tfjs-node`.

---

Desarrollado por: [Tu equipo de IA Fitness] 💪
