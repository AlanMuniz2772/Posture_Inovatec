# ============================================
# ARCHIVO: app_squat.py
# OBJETIVO: Aplicación web que analiza peso muerto en tiempo real con IA
# TECNOLOGÍAS: Dash + Flask + MediaPipe + TensorFlow + OpenCV
# ============================================

import mediapipe as mp
import SquatPosture as sp
import cv2
import tensorflow as tf
import numpy as np
import time
import utils
import os

# Inicialización de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Carpeta donde están los videos
VIDEO_DIR = "./data_processing/videos_recortados"

# Recorre todos los archivos de la carpeta
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

# Cargar modelo multietiqueta entrenado
model = tf.keras.models.load_model("modelo2.keras")

pose = mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
# === PROCESAMIENTO DE VIDEO ===

cap = cv2.VideoCapture(1) 

sp.show_prediction(cap, pose, model)

cap.release()
cv2.destroyAllWindows()


