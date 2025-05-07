import cv2
import mediapipe as mp
import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils as utils

# Rutas
VIDEO_DIR = "./videos_recortados"
OUTPUT_JSON = "./data/input_vectors.json"

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Resultado final
output_data = []

def save_json():
    # Guardar en JSON
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ input_vectors.json generado con {len(output_data)} frames.")

try:
    # Procesar cada video
    for video_file in os.listdir(VIDEO_DIR):
        if not video_file.endswith(".mp4"):
            continue

        print(f"📽️ Procesando: {video_file}")
        cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video_file))
        frame_num = 0

        while cap.isOpened():
            image, results = utils.get_frame(cap, pose, mp_pose)

            if results.pose_landmarks:
                # Extraer los 99 valores (x, y, z de cada punto)
                landmarks = [
                    coord
                    for lm in results.pose_landmarks.landmark
                    for coord in (lm.x, lm.y, lm.z)
                ]

                output_data.append({
                    "video": video_file,
                    "frame": frame_num,
                    "landmarks": landmarks
                })

            frame_num += 1
            cv2.imshow("Landmarks", image)
        cap.release()
        save_json()
except Exception as e:
    print(e)
    save_json()
    print(f"❌ Error procesando el video: {video_file}")


save_json()

print(f"✅ input_vectors.json generado con {len(output_data)} frames.")
