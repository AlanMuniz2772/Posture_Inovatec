# ============================================
# ARCHIVO: utils.py
# OBJETIVO: Funciones de soporte visual y auditivo para feedback de postura
# ============================================

import cv2                  # OpenCV para manipulación visual de imágenes
import numpy as np          # NumPy para cálculos numéricos
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils


# ============================================
# FUNCIÓN: landmarks_list_to_array
# OBJETIVO: Convertir landmarks normalizados a coordenadas reales en pixeles
# ============================================
def landmarks_list_to_array(landmark_list, image_shape):
    rows, cols, _ = image_shape

    if landmark_list is None:
        return None

    # Multiplica los puntos normalizados por el tamaño real de la imagen
    return np.asarray([
        (lmk.x * cols, lmk.y * rows)
        for lmk in landmark_list.landmark
    ])

# ============================================
# FUNCIÓN: label_params
# OBJETIVO: Dibujar los valores de los 5 parámetros clave sobre el frame
# ============================================
def label_params(frame, params, coords):
    if coords is None:
        return

    # Convertir de radianes a grados para mostrar
    params = params * 180 / np.pi

    # Coordenadas medias para cada parte del cuerpo donde se etiqueta el ángulo
    neck = (coords[11] + coords[12]) / 2
    cv2.putText(frame, str(np.round(params[0], 2)), (int(neck[0]), int(neck[1]) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    knee = (coords[25] + coords[26]) / 2
    cv2.putText(frame, str(np.round(params[1], 2)), (int(knee[0]), int(knee[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    hip = (coords[23] + coords[24]) / 2
    cv2.putText(frame, str(np.round(params[2], 2)), (int(hip[0]), int(hip[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    ankle = (coords[27] + coords[28]) / 2
    cv2.putText(frame, str(np.round(params[3], 2)), (int(ankle[0]), int(ankle[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    y_knee = (coords[25] + coords[26]) / 2
    cv2.putText(frame, str(np.round(params[4], 2)), (int(y_knee[0]), int(y_knee[1]) - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# ============================================
# FUNCIÓN: label_final_results
# OBJETIVO: Mostrar feedback textual y verbal en la imagen final
# ============================================
def label_final_results(image, label):
    # Diccionario para traducir códigos de corrección a mensajes explicativos
    expanded_labels = {
        "c": "Correct Form",
        "k": "Knee Ahead, push your butt out",
        "h": "Back Wrongly Positioned, keep your chest up",
        "r": "Back Wrongly Positioned, keep your chest up",
        "x": "Correct Depth"
    }

    image_width, image_height, _ = image.shape

    # Separar el string de etiquetas en caracteres individuales
    label_list = [char for char in label]
    described_label = list(map(lambda x: expanded_labels[x], label_list))

    # Determinar color del rectángulo: verde si es correcto, azul si hay errores
    color = (42, 210, 48) if "c" in label_list else (13, 13, 205)

    # Dibujar una caja de fondo en la parte superior de la imagen
    cv2.rectangle(image, (0, 0), (image_height, 74), color, -1)

    # Crear el mensaje a mostrar y decir
    instruction = "   " + " + ".join(described_label)


    # Mostrar el texto en pantalla
    cv2.putText(image, instruction, (0, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def get_frame(cap, pose, mp_pose):
    ret, frame = cap.read()
    if not ret:
        return None

    # Convertir a RGB y procesar con MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # Dibujar landmarks sobre el frame si se detectan
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=1)
        )


    frame = cv2.resize(frame, (720, 480))
    return frame