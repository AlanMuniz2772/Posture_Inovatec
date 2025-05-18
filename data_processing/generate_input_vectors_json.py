import cv2
import mediapipe as mp
import json
import os

# Configuración
VIDEO_FOLDER = "./videos_recortados"
OUTPUT_FILE = "./input_vectors_2.jsonl"


mp_pose = mp.solutions.pose

pose = mp_pose.Pose(static_image_mode=False,
                  model_complexity=2,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5)

landmark_enum = mp_pose.PoseLandmark

with open(OUTPUT_FILE, "w") as fout:
    for index, filename in enumerate(os.listdir(VIDEO_FOLDER)):
        if not filename.endswith(".mp4"):
            continue

        video_path = os.path.join(VIDEO_FOLDER, filename)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        print(f"🎥 Procesando {index} - {filename}...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Procesar frame con MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Diccionario nombrado de landmarks con filtro de visibilidad
                named_landmarks = {
                    lm.name: (
                        [
                            round(landmarks[lm.value].x, 5),
                            round(landmarks[lm.value].y, 5),
                            round(landmarks[lm.value].z, 5)
                        ] if landmarks[lm.value].visibility >= 0.5 else None
                    )
                    for lm in landmark_enum
                }

                # Guardar entrada como línea JSON
                data = {
                    "index":index,
                    "video": filename,
                    "frame": frame_count,
                    "landmarks": named_landmarks
                }
                fout.write(json.dumps(data) + "\n")

            frame_count += 1

        cap.release()

print("✅ input_vectors_2.jsonl generado correctamente.")
