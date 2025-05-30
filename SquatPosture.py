import cv2
import mediapipe as mp
import math
import numpy as np
import utils
import pyttsx3
import threading
import time

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def hablar_async(texto):
    threading.Thread(target=hablar, args=(texto,), daemon=True).start()

def hablar(texto):
    if not engine._inLoop:
        engine.say(texto)
        engine.runAndWait()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

LANDMARKS_NECESARIOS = [
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_HIP", "RIGHT_HIP"
]

EXERCISES = [
    'squats',
    'planks',
]

def radian_to_degrees(radian):
    return radian * 180 / math.pi

def get_angle(v1, v2):
    dot = np.dot(v1, v2)
    mod_v1 = np.linalg.norm(v1)
    mod_v2 = np.linalg.norm(v2)
    cos_theta = dot/(mod_v1*mod_v2)
    theta = math.acos(cos_theta)
    return theta

def get_length(v):
    return np.linalg.norm(v)


def get_params(results, exercise='squats', all=False):

    if results.pose_landmarks is None:
        if exercise == 'squats':
            return np.zeros((1, 5) if not all else (19,3))
        else:
            return np.array([0, 0])

    points = {}
    nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    points["NOSE"] = np.array([nose.x, nose.y, nose.z])
    left_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
    points["LEFT_EYE"] = np.array([left_eye.x, left_eye.y, left_eye.z])
    right_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
    points["RIGHT_EYE"] = np.array([right_eye.x, right_eye.y, right_eye.z])
    mouth_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]
    points["MOUTH_LEFT"] = np.array([mouth_left.x, mouth_left.y, mouth_left.z])
    mouth_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]
    points["MOUTH_RIGHT"] = np.array([mouth_right.x, mouth_right.y, mouth_right.z])
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    points["LEFT_SHOULDER"] = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    points["RIGHT_SHOULDER"] = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])
    left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    points["LEFT_ELBOW"] = np.array([left_elbow.x, left_elbow.y, left_elbow.z])
    right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    points["RIGHT_ELBOW"] = np.array([right_elbow.x, right_elbow.y, right_elbow.z])
    right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    points["RIGHT_WRIST"] = np.array([right_wrist.x, right_wrist.y, right_wrist.z])
    left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    points["LEFT_WRIST"] = np.array([left_wrist.x, left_wrist.y, left_wrist.z])
    left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    points["LEFT_HIP"] = np.array([left_hip.x, left_hip.y, left_hip.z])
    right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    points["RIGHT_HIP"] = np.array([right_hip.x, right_hip.y, right_hip.z])
    left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    points["LEFT_KNEE"] = np.array([left_knee.x, left_knee.y, left_knee.z])
    right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    points["RIGHT_KNEE"] = np.array([right_knee.x, right_knee.y, right_knee.z])
    left_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
    points["LEFT_HEEL"] = np.array([left_heel.x, left_heel.y, left_heel.z])
    right_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
    points["RIGHT_HEEL"] = np.array([right_heel.x, right_heel.y, right_heel.z])
    left_foot_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
    points["LEFT_FOOT_INDEX"] = np.array([left_foot_index.x, left_foot_index.y, left_foot_index.z])
    right_foot_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
    points["RIGHT_FOOT_INDEX"] = np.array([right_foot_index.x, right_foot_index.y, right_foot_index.z])
    left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    points["LEFT_ANKLE"] = np.array([left_ankle.x, left_ankle.y, left_ankle.z])
    right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    points["RIGHT_ANKLE"] = np.array([right_ankle.x, right_ankle.y, right_ankle.z])

    points["MID_SHOULDER"] = (points["LEFT_SHOULDER"] + points["RIGHT_SHOULDER"]) / 2
    points["MID_HIP"] = (points["LEFT_HIP"] + points["RIGHT_HIP"]) / 2

    z_eyes = (points["RIGHT_EYE"][2] + points["LEFT_EYE"][2]) / 2
    z_mouth = (points["MOUTH_LEFT"][2] + points["MOUTH_RIGHT"][2]) / 2

    theta_neck = get_angle(np.array([0, 0, -1]),
                           points["NOSE"] - points["MID_HIP"])

    theta_s1 = get_angle(points["LEFT_ELBOW"]-points["LEFT_SHOULDER"],
                         points["LEFT_HIP"]-points["LEFT_SHOULDER"])

    theta_s2 = get_angle(points["RIGHT_ELBOW"] - points["RIGHT_SHOULDER"],
                         points["RIGHT_HIP"] - points["RIGHT_SHOULDER"])

    theta_s = (theta_s1 + theta_s2) / 2

    z_face = z_eyes - z_mouth

    theta_k1 = get_angle(points["RIGHT_HIP"] - points["RIGHT_KNEE"],
                         points["RIGHT_ANKLE"] - points["RIGHT_KNEE"])

    theta_k2 = get_angle(points["LEFT_HIP"] - points["LEFT_KNEE"],
                         points["LEFT_ANKLE"] - points["LEFT_KNEE"])

    theta_k = (theta_k1 + theta_k2) / 2

    theta_h1 = get_angle(points["RIGHT_KNEE"] - points["RIGHT_HIP"],
                         points["RIGHT_SHOULDER"] - points["RIGHT_HIP"])

    theta_h2 = get_angle(points["LEFT_KNEE"] - points["LEFT_HIP"],
                         points["LEFT_SHOULDER"] - points["LEFT_HIP"])

    theta_h = (theta_h1 + theta_h2) / 2

    torso_length = get_length(points['MID_SHOULDER'] - points['MID_HIP'])
    left_thigh_length = get_length(points['LEFT_KNEE'] - points['LEFT_HIP'])
    right_thigh_length = get_length(points['RIGHT_KNEE'] - points['RIGHT_HIP'])
    left_tibula_length = get_length(points['LEFT_KNEE'] - points['LEFT_HEEL'])
    right_tibula_length = get_length(points['RIGHT_KNEE'] - points['RIGHT_HEEL'])

    thigh_length = (left_thigh_length + right_thigh_length) / 2
    tibula_length = (left_tibula_length + right_tibula_length) / 2

    length_normalization_factor = (1 / (tibula_length))**0.5

    z1 = (points["RIGHT_ANKLE"][2] + points["RIGHT_HEEL"][2]) / 2 - points["RIGHT_FOOT_INDEX"][2]

    z2 = (points["LEFT_ANKLE"][2] + points["LEFT_HEEL"][2]) / 2 - points["LEFT_FOOT_INDEX"][2]

    z = (z1 + z2) / 2

    z *= length_normalization_factor

    left_foot_y = (points["LEFT_ANKLE"][1] + points["LEFT_HEEL"][1] + points["LEFT_FOOT_INDEX"][1]) / 3
    right_foot_y = (points["RIGHT_ANKLE"][1] + points["RIGHT_HEEL"][1] + points["RIGHT_FOOT_INDEX"][1]) / 3

    left_ky = points["LEFT_KNEE"][1] - left_foot_y
    right_ky = points["RIGHT_KNEE"][1] - right_foot_y

    ky = (left_ky + right_ky) / 2

    ky *= length_normalization_factor

    left_foot = points["LEFT_HEEL"] - points["LEFT_FOOT_INDEX"]
    theta_left_foot = get_angle(left_foot, np.array([left_foot[0], left_foot[1], points["LEFT_FOOT_INDEX"][2]]))
    right_foot = points["RIGHT_HEEL"] - points["RIGHT_FOOT_INDEX"]
    theta_right_foot = get_angle(right_foot, np.array([right_foot[0], right_foot[1], points["RIGHT_FOOT_INDEX"][2]]))

    theta_foot = (theta_right_foot + theta_left_foot) / 2

    if exercise=='squats':
        params = np.array([theta_neck, theta_k, theta_h, z, ky])
    elif exercise=='plank':
        params = np.array([theta_s1, theta_s2])

    if all:
        params = np.array([[x, y, z] for pos, (x, y, z) in points.items()]) * length_normalization_factor

    return np.round(params, 2)


def calcular_parametros_desde_resultados(points):
    if not points:
        # Retorna vector neutral si no hay detección
        return np.zeros(5)

    # Puntos medios

    points["MID_SHOULDER"] = (points["LEFT_SHOULDER"] + points["RIGHT_SHOULDER"]) / 2
    points["MID_HIP"] = (points["LEFT_HIP"] + points["RIGHT_HIP"]) / 2

    # Ángulo del cuello (usando nariz - cadera)
    v_torso = points["MID_SHOULDER"]- points["MID_HIP"]
    theta_neck = get_angle(np.array([0, -1, 0]), v_torso)

    # Ángulo rodilla
    theta_k1 = get_angle(points["RIGHT_HIP"] - points["RIGHT_KNEE"], points["RIGHT_ANKLE"] - points["RIGHT_KNEE"])
    theta_k2 = get_angle(points["LEFT_HIP"] - points["LEFT_KNEE"], points["LEFT_ANKLE"] - points["LEFT_KNEE"])
    theta_k = (theta_k1 + theta_k2) / 2

    # Ángulo cadera
    theta_h1 = get_angle(points["RIGHT_KNEE"] - points["RIGHT_HIP"], points["RIGHT_SHOULDER"] - points["RIGHT_HIP"])
    theta_h2 = get_angle(points["LEFT_KNEE"] - points["LEFT_HIP"], points["LEFT_SHOULDER"] - points["LEFT_HIP"])
    theta_h = (theta_h1 + theta_h2) / 2

    # Normalización por tibia
    tibula_R = get_length(points["RIGHT_KNEE"] - points["RIGHT_HEEL"])
    tibula_L = get_length(points["LEFT_KNEE"] - points["LEFT_HEEL"])
    tibula_avg = (tibula_R + tibula_L) / 2
    norm_factor = (1 / tibula_avg) ** 0.5

    # Z: profundidad relativa del pie
    z1 = (points["RIGHT_ANKLE"][2] + points["RIGHT_HEEL"][2]) / 2 - points["RIGHT_FOOT_INDEX"][2]
    z2 = (points["LEFT_ANKLE"][2] + points["LEFT_HEEL"][2]) / 2 - points["LEFT_FOOT_INDEX"][2]
    z = (z1 + z2) / 2 * norm_factor

    # KY: altura de rodillas sobre el suelo
    y_knee_L = np.mean([points["LEFT_ANKLE"][1], points["LEFT_HEEL"][1], points["LEFT_FOOT_INDEX"][1]]) - points["LEFT_KNEE"][1]
    y_knee_R = np.mean([points["RIGHT_ANKLE"][1], points["RIGHT_HEEL"][1], points["RIGHT_FOOT_INDEX"][1]]) - points["RIGHT_KNEE"][1]

    ky = (y_knee_L + y_knee_R) / 2 * norm_factor
    
    params = np.array([theta_neck, theta_k, theta_h, z, ky])
    # Convertir a grados los ángulos y devolver
    return np.round(params, 2)


def calcular_deadlift(points):
    if not points or all(p is None for p in points.values()):
        return None
    
    return {
        "pies_a_la_anchura_de_hombros": pies_a_la_anchura_de_hombros(points),
        "agarre_amplio": agarre_amplio_manos_fuera_de_las_piernas(points),
        "espalda_neutral": espalda_en_posicion_neutral(points),
        "hombros_sobre_barra": hombros_sobre_la_barra(points)
    }
    
    



def pies_a_la_anchura_de_hombros(points, tolerancia=0.3):
    """
    Evalúa si los tobillos están separados aproximadamente al mismo ancho que los hombros.

    Parámetros:
    - points: dict con los puntos clave nombrados, como {'LEFT_ANKLE': np.array([x, y, z]), ...}
    - tolerancia: porcentaje de tolerancia aceptable (por ejemplo, 0.2 = ±20%)

    Retorna:
    - True si la distancia de los tobillos es similar a la de los hombros
    - False si no
    - None si falta algún punto
    """
    claves = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ANKLE', 'RIGHT_ANKLE']
    if any(points.get(k) is None for k in claves):
        return None  # No se puede evaluar

    
    def distancia_x(p1, p2):
        return abs(p1[0] - p2[0])

    ancho_hombros = distancia_x(points['LEFT_SHOULDER'], points['RIGHT_SHOULDER'])
    ancho_pies = distancia_x(points['LEFT_ANKLE'], points['RIGHT_ANKLE'])

    margen = ancho_hombros * tolerancia
    return (ancho_hombros - margen) <= ancho_pies <= (ancho_hombros + margen)


def agarre_amplio_manos_fuera_de_las_piernas(points, margen_extra=0.02):
    """
    Evalúa si el agarre es amplio, es decir, si las manos (muñecas) están fuera del ancho de las rodillas.
    
    Parámetros:
    - points: dict con los landmarks nombrados (np.array o None)
    - margen_extra: cantidad mínima adicional que las muñecas deben exceder sobre las rodillas (en proporción)

    Retorna:
    - True si el agarre es amplio
    - False si no lo es
    - None si faltan puntos clave
    """
    claves = ['LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_KNEE', 'RIGHT_KNEE']
    if any(points.get(k) is None for k in claves):
        return None

    def distancia_x(p1, p2):
        return abs(p1[0] - p2[0])

    ancho_rodillas = distancia_x(points['LEFT_KNEE'], points['RIGHT_KNEE'])
    ancho_manos = distancia_x(points['LEFT_WRIST'], points['RIGHT_WRIST'])

    return ancho_manos >= (ancho_rodillas + margen_extra)




def espalda_en_posicion_neutral(points, tolerancia_grados=50):
    """
    Evalúa si la espalda está en una posición neutral observando la inclinación del tronco
    desde el hombro a la cadera en ambos lados del cuerpo.
    
    Parámetros:
    - points: dict con los landmarks nombrados (np.array o None)
    - tolerancia_grados: margen angular permitido con respecto a la vertical

    Retorna:
    - True si ambos lados del tronco están dentro del rango de neutralidad
    - False si la inclinación es excesiva
    - None si faltan puntos clave
    """
    claves = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']
    if any(points.get(k) is None for k in claves):
        return None

    def angulo_con_vertical(p1, p2):
        vector = p2 - p1  # de hombro a cadera
        vertical = np.array([0, 1, 0])  # eje Y
        cos_theta = np.dot(vector[:2], vertical[:2]) / (np.linalg.norm(vector[:2]) * np.linalg.norm(vertical[:2]))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # estabilidad numérica
        angulo = np.degrees(np.arccos(cos_theta))
        return angulo

    angulo_izq = angulo_con_vertical(points['LEFT_SHOULDER'], points['LEFT_HIP'])
    angulo_der = angulo_con_vertical(points['RIGHT_SHOULDER'], points['RIGHT_HIP'])

    angulo_promedio = (angulo_izq + angulo_der) / 2

    return abs(angulo_promedio) <= tolerancia_grados



def hombros_sobre_la_barra(points, tolerancia_x=0.05):
    """
    Evalúa si los hombros están aproximadamente sobre las muñecas (la barra) en vista lateral.
    
    Parámetros:
    - points: dict con np.array o None
    - tolerancia_x: rango aceptable de diferencia horizontal (X)

    Retorna:
    - True si ambos hombros están sobre la barra
    - False si no lo están
    - None si faltan puntos clave
    """
    claves = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_WRIST', 'RIGHT_WRIST']
    if any(points.get(k) is None for k in claves):
        return None

    def alineacion_horizontal(hombro, muñeca):
        diferencia = abs(hombro[0] - muñeca[0])
        return diferencia <= tolerancia_x

    izq = alineacion_horizontal(points['LEFT_SHOULDER'], points['LEFT_WRIST'])
    der = alineacion_horizontal(points['RIGHT_SHOULDER'], points['RIGHT_WRIST'])

    return izq and der


def auto_label(params):
    theta_neck, theta_k, theta_h, z, ky = params
    labels = []
    
    # Convertir a grados
    theta_neck_deg = math.degrees(theta_neck)
    theta_k_deg = math.degrees(theta_k)
    theta_h_deg = math.degrees(theta_h)

    # Reglas relajadas
    if theta_k_deg < 80:
        labels.append("k")

    if theta_h_deg < 50:
        labels.append("h")

    if theta_neck_deg > 45:
        labels.append("r")

    if 0.15 <= ky <= 0.35:
        labels.append("x")

    if z > 0.2:
        labels.append("i")

    if not labels or labels == ["x"]:
        labels.append("c")


    return labels


def obtener_vector_para_modelo(results, threshold=0.5):
    """
    Devuelve un vector plano de 30 valores (10 landmarks x 3 coords),
    usando -1.0 como valor para landmarks con visibilidad baja o ausentes.
    """
    if results is None:
        # return np.array([-1.0] * 30)  # Todos los landmarks faltantes
        return None

    landmark_list = results.pose_landmarks.landmark
    landmark_enum = mp.solutions.pose.PoseLandmark

    vector = []
    for nombre in LANDMARKS_NECESARIOS:
        idx = landmark_enum[nombre].value
        punto = landmark_list[idx]

        if punto.visibility < threshold:
            # vector.extend([-1.0, -1.0, -1.0])
            return None  # Landmark no visible
        else:
            vector.extend([punto.x, punto.y, punto.z])

    return np.array(vector)


def show_prediction(cap, pose, model):
    mensaje = None
    
    while cap.isOpened():
        image, results = utils.get_frame(cap, pose, mp_pose)

        if image is None or image.shape[0] == 0 or image.shape[1] == 0:
            print("⚠️ Frame inválido. Se omite.")
            break

        nuevo_mensaje = None
        

        if results and results.pose_landmarks:
            landmarks = obtener_vector_para_modelo(results)
            if landmarks is not None:
                output = model.predict(np.array([landmarks]))[0]
                image, nuevo_mensaje = utils.label_final_results(image, output)
            else:
                nuevo_mensaje = "No se detectaron landmarks suficientes"

            if nuevo_mensaje is not None and nuevo_mensaje != mensaje:
                hablar_async(nuevo_mensaje)
                mensaje = nuevo_mensaje
                
        cv2.imshow("Landmarks", image)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

# if __name__ == "__main__":
#     print(radian_to_degrees(3))