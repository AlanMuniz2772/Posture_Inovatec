import cv2
import mediapipe as mp
import os
import time as time
import sys
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils as utils
import SquatPosture as sp

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose

# Carpeta donde están los videos
VIDEO_DIR = "./videos_recortados"

# Recorre todos los archivos de la carpeta
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

pose = mp_pose.Pose(static_image_mode=False,
                  model_complexity=2,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5)

for video_file in video_files:
    print(f"\n Procesando video: {video_file}")
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video_file))

    while cap.isOpened():
        image, results = utils.get_frame(cap, pose, mp_pose)

        if image is None:
            break
        

        if results.pose_landmarks:
            points = utils.get_points(results)
            
            params = sp.calcular_deadlift(points)
            print("HOMBROS_PIES:", params[0], "MANOS_PIERNAS:", params[1], "ESPALDA:", params[2], "HOMBROS_BARRA:", params[3])

            # coords = utils.landmarks_list_to_array(results.pose_landmarks, image.shape)

            # utils.label_params(image, params, coords)

            # if params[3] > 1:
            # time.sleep(1)

        # Mostrar el frame con los puntos
        cv2.imshow("Landmarks", image)
        
        # Esperar (presiona 'q' para salir del video)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
