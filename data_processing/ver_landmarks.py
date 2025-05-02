import cv2
import mediapipe as mp
import os
import time
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils as utils

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose

# Carpeta donde están los videos
VIDEO_DIR = "./videos"

# Recorre todos los archivos de la carpeta
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

with mp_pose.Pose(static_image_mode=False,
                  model_complexity=1,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    for video_file in video_files:
        print(f"\n Procesando video: {video_file}")
        cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video_file))

        while cap.isOpened():
            frame = utils.get_frame(cap, pose, mp_pose)

            if frame is None:
                break

            # Mostrar el frame con los puntos
            cv2.imshow("Landmarks", frame)

            # Esperar (presiona 'q' para salir del video)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
