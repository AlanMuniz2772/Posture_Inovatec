# ============================
# ARCHIVO: generate_input_vectors.py
# OBJETIVO: Procesar videos de sentadillas para extraer vectores de entrada con MediaPipe
# SALIDA: input_vectors.csv con parámetros posturales por frame
# ============================

import cv2                              # Para procesamiento de video
import mediapipe as mp                 # Para detección de poses humanas   # Módulo personalizado con funciones específicas para sentadillas
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils as utils
import SquatPosture as sp
           

# Inicialización de utilidades de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ========== BLOQUE PRINCIPAL ==========
if __name__ == '__main__':
    # Ruta donde están los videos ya procesados
    VIDEO_DIR = './videos_recortados'


    # Abrir el archivo de salida donde se guardarán los vectores
    file = open("./input_vectors_deadlift.csv", "w")
    
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

    with mp_pose.Pose(static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
        # Procesar cada video en la carpeta
        for index,  video_name in enumerate(video_files):
            # Cargar video
            cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video_name))
            frame_number = 0  # Contador de frames

            # Iniciar el modelo de detección de pose con MediaPipe
            
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    break  # Salir si no hay más frames

                # Convertir imagen a RGB y deshabilitar escritura para procesarla
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)  # Detección de pose


                # Obtener los parámetros relevantes usando función personalizada
                params = sp.calcular_parametros_desde_resultados(results)


                # Escribir datos al archivo CSV (nombre del video, frame y 5 parámetros)
                file.write("{},{},{},{},{},{},{}\n".format(
                    video_name,               # ID del video (primeros 3 caracteres)
                    frame_number + 1,              # Número de frame (desde 1)
                    params[0], params[1], params[2],
                    params[3], params[4]
                ))
                file.flush()  # Forzar escritura inmediata al disco

                frame_number += 1



            # Liberar recursos del video al terminar
            cap.release()
            print(index, video_name)  # Imprimir nombre del video procesado

    # Cerrar archivo CSV y destruir ventanas de OpenCV
    file.close()

