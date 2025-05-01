# ============================================
# ARCHIVO: app_squat.py
# OBJETIVO: Aplicación web interactiva que analiza sentadillas en tiempo real con IA
# TECNOLOGÍAS: Dash + Flask + MediaPipe + TensorFlow + OpenCV
# ============================================

import dash                       # Framework web basado en Flask
from dash import dcc, html        # Componentes HTML de Dash
import mediapipe as mp            # Para detección de postura humana
import SquatPosture as sp         # Análisis de postura personalizado
from flask import Flask, Response # Servidor Flask para video streaming
import cv2                        # Para acceso a la cámara y manipulación de video
import tensorflow as tf           # Modelo entrenado para clasificar errores posturales
import numpy as np
from utils import landmarks_list_to_array, label_params, label_final_results  # Funciones de visualización
from keras.layers import TFSMLayer  # (No utilizado directamente aquí)

# Inicialización de utilidades de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Cargar modelo entrenado
model = tf.keras.models.load_model("working_model_1.keras")

# ============================================
# CLASE: VideoCamera
# OBJETIVO: Encapsular acceso a la cámara web
# ============================================
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(2)  # Acceder a cámara 0 (por defecto)

    def __del__(self):
        self.video.release()  # Liberar cámara al destruir el objeto

# ============================================
# FUNCIÓN: gen
# OBJETIVO: Proceso de lectura, análisis y etiquetado de video en vivo
# ============================================
def gen(camera):
    cap = camera.video
    i = 0
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            image = cv2.flip(image, 1)  # Voltear para que sea como un espejo

            if not success:
                print("Ignoring empty camera frame.")
                break

            # Convertir imagen a RGB y reducir tamaño para análisis más rápido
            image_height, image_width, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dim = (image_width // 5, image_height // 5)
            resized_image = cv2.resize(image_rgb, dim)

            # Procesar pose
            results = pose.process(resized_image)

            # Obtener parámetros biomecánicos
            params = sp.get_params(results)
            flat_params = np.reshape(params, (5, 1))

            # Mostrar landmarks sobre la imagen original
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=1)
                )
            # Pasar parámetros por el modelo para predecir errores
            output = model.predict(flat_params.T)

            # Reescalar valores para ajustar pesos de errores
            output[0][0] *= 0.7  # 'c' - correcto
            output[0][1] *= 1.7  # 'k' - rodillas
            output[0][2] *= 4    # 'h' - espalda
            output[0][3] *= 0    # 'r' - (no usado o ignorado)
            output[0][4] *= 5    # 'x' - profundidad

            # Normalizar probabilidades
            output = output * (1 / np.sum(output))
            output_name = ['c', 'k', 'h', 'r', 'x', 'i']

            # Ajuste específico para exagerar la espalda (¿debug?)
            output[0][2] += 0.1

            # Crear etiqueta final
            label = ""
            for i in range(1, 4):  # k, h, r
                if output[0][i] > 0.5:
                    label += output_name[i]
            if label == "":
                label = "c"
            label += 'x' if output[0][4] > 0.15 and label == 'c' else ''

            # Mostrar texto sobre la imagen y hablarlo
            label_final_results(image, label)

            # Codificar imagen como JPEG para streaming
            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# ============================================
# SERVIDOR Y APLICACIÓN DASH
# ============================================
server = Flask(__name__)
app = dash.Dash(__name__, server=server)
app.title = "Posture"

# Ruta para alimentar video en vivo
@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

# Diseño visual de la app
app.layout = html.Div(className="main", children=[
    html.Link(
        rel="stylesheet",
        href="/assets/stylesheet.css"  # Cargar estilos personalizados
    ),
    dcc.Markdown(
        children="""
        <div class="main-container">
            <table cellspacing="20px" class="table">
                <tr class="row">
                    <td> <img src="/assets/animation_for_web.gif" class="logo" /> </td>
                </tr>
                <tr class="choices">
                    <td> Your personal AI Gym Trainer </td>
                </tr>
                <tr class="row">
                    <td> <img src="/video_feed" class="feed"/> </td>
                </tr>
                <tr class="disclaimer">
                    <td> Please ensure that the scene is well lit and your entire body is visible </td>
                </tr>
            </table>
        </div>
        """,
        dangerously_allow_html=True  # Permite HTML puro dentro del Markdown
    ),
])

# ============================================
# INICIO DE LA APLICACIÓN
# ============================================
if __name__ == '__main__':
    app.run(debug=True)
