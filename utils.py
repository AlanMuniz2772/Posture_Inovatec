# ============================================
# ARCHIVO: utils.py
# OBJETIVO: Funciones de soporte visual y auditivo para feedback de postura
# ============================================

import cv2                  # OpenCV para manipulación visual de imágenes
import numpy as np          # NumPy para cálculos numéricos
import mediapipe as mp
import time

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

# === FUNCION PARA ETIQUETAR RESULTADOS ===
def label_final_results(image, output, threshold=0.5):

    mensajes = ("Alinea tus pies a la anchura de hombros",
               "Las manos deben esatr ams anchas que las piernas", 
               "Inclina menos tu espalda",
               "Alinea hombros con muñecas")

    index = 0
    biggest = 0

    output[0] *= 0.75
    output[2] *= 10
    output[3] *= 0.6
    
    
    for i, r  in enumerate(output):  
        if r > biggest:
            biggest = r
            index = i

    mensaje = "Todo bien"

    if biggest > threshold:
        mensaje = mensajes[index]
        color = (13, 13, 205)
        # if index == 3:
        #     print("hombros-muñecas", output[index])
        #     time.sleep(1)
    else:
        color = (42, 210, 48)


    image_height, _, _ = image.shape
    cv2.rectangle(image, (0, 0), (image_height, 74), color, -1)
    cv2.putText(image, mensaje, (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image, mensaje



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

    image = cv2.resize(image, (720, 480))
    
    return image, results


def get_points(results):
    landmark_list = results.pose_landmarks.landmark

    # Diccionario nombrado de landmarks con filtro de visibilidad
    points = {
        lm.name: (
            np.array([
                landmark_list[lm.value].x,
                landmark_list[lm.value].y,
                landmark_list[lm.value].z
            ]) if landmark_list[lm.value].visibility >= 0.5 else None  # <-- sin np.array(None)
        )
        for lm in mp.solutions.pose.PoseLandmark
    }

    return points