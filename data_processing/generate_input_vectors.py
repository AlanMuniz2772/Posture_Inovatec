# ============================
# ARCHIVO: generate_input_vectors.py
# OBJETIVO: Procesar videos de sentadillas para extraer vectores de entrada con MediaPipe
# SALIDA: input_vectors.csv con parámetros posturales por frame
# ============================

import cv2                              # Para procesamiento de video
import mediapipe as mp                 # Para detección de poses humanas
import Posture.SquatPosture as sp      # Módulo personalizado con funciones específicas para sentadillas
import numpy as np
import os
from Posture.utils import *            # Funciones auxiliares para graficar y procesar landmarks

# Inicialización de utilidades de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ========== BLOQUE PRINCIPAL ==========
if __name__ == '__main__':
    # Ruta donde están los videos ya procesados
    directory = './data/processed'

    # Obtener y ordenar los nombres de los archivos de video
    video_names = sorted(os.listdir(directory))

    # Abrir el archivo de salida donde se guardarán los vectores
    file = open("./data/input_vectors.csv", "w")

    # Procesar cada video en la carpeta
    for video_name in video_names:
        # Cargar video
        cap = cv2.VideoCapture(os.path.join(directory, video_name))
        frame_number = 0  # Contador de frames

        # Iniciar el modelo de detección de pose con MediaPipe
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    break  # Salir si no hay más frames

                # Convertir imagen a RGB y deshabilitar escritura para procesarla
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)  # Detección de pose

                # Habilitar escritura nuevamente y regresar imagen a BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Obtener alto y ancho de la imagen
                image_height, image_width, _ = image.shape

                # Obtener los parámetros relevantes usando función personalizada
                params = sp.get_params(results)

                # Dibujar los landmarks detectados sobre la imagen
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                # Convertir los landmarks a coordenadas absolutas
                coords = landmarks_list_to_array(results.pose_landmarks, image.shape)

                # Etiquetar visualmente los parámetros sobre la imagen
                label_params(image, params, coords)

                # Escribir datos al archivo CSV (nombre del video, frame y 5 parámetros)
                file.write("{},{},{},{},{},{},{}\n".format(
                    video_name[0:3],               # ID del video (primeros 3 caracteres)
                    frame_number + 1,              # Número de frame (desde 1)
                    params[0], params[1], params[2],
                    params[3], params[4]
                ))
                file.flush()  # Forzar escritura inmediata al disco

                frame_number += 1

                # Mostrar imagen procesada con pose
                cv2.imshow('MediaPipe Pose', image)

                # Salir si se presiona la tecla ESC (código ASCII 27)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        # Liberar recursos del video al terminar
        cap.release()
        print(video_name)  # Imprimir nombre del video procesado

    # Cerrar archivo CSV y destruir ventanas de OpenCV
    file.close()
    cv2.destroyAllWindows()
