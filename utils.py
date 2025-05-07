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
    if coords is None or params is None:
        return

    # Coordenadas de referencia: 
    # params[0] = theta_neck, [1] = theta_k, [2] = theta_h, [3] = z, [4] = ky

    # θ_neck → texto entre hombros
    neck = (coords[11] + coords[12]) / 2
    cv2.putText(frame, f"Neck: {params[0]:.2f}", (int(neck[0]), int(neck[1]) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # θ_k (rodilla)
    knee = (coords[25] + coords[26]) / 2
    cv2.putText(frame, f"K: {params[1]:.2f}", (int(knee[0]), int(knee[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # θ_h (cadera)
    hip = (coords[23] + coords[24]) / 2
    cv2.putText(frame, f"H: {params[2]:.2f}", (int(hip[0]), int(hip[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Z (profundidad de pie)
    ankle = (coords[27] + coords[28]) / 2
    cv2.putText(frame, f"Z: {params[3]:.3f}", (int(ankle[0]), int(ankle[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    # KY (altura rodilla)
    y_knee = (coords[25] + coords[26]) / 2
    cv2.putText(frame, f"ky: {params[4]:.3f}", (int(y_knee[0]), int(y_knee[1]) - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


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
    "r": "Neck Misaligned, keep your neck neutral",
    "x": "Correct Depth",
    "i": "Foot instability detected"
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
        return None, None

    # Convertir a RGB y procesar con MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Dibujar landmarks sobre el frame si se detectan
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
        )

    # frame = cv2.resize(frame, (720, 480))
    
    return image, results


