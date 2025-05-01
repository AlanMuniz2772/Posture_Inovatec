# ============================================
# ARCHIVO: generate_output_vectors.py
# OBJETIVO: Generar vectores de salida etiquetados por frame, 
#           a partir de un archivo con etiquetas por rango de tiempo
# SALIDA: output_vectors.csv
# ============================================

import csv          # Para leer archivos CSV
import cv2          # Para contar frames de los videos

# ============================================
# FUNCIÓN AUXILIAR: Obtener total de frames de un video
# ============================================
def get_total_frames(video):
    filename = "data/processed/" + video + "_squat.mp4"
    cap = cv2.VideoCapture(filename)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames

# Lista donde se guardarán las filas del archivo de etiquetas
rows = []

# Se asume que los videos tienen 12 frames por segundo
fps = 12

# ============================================
# CARGAR ARCHIVO labels.csv
# Formato esperado por línea: [video_id tiempo etiquetas]
# Ejemplo: "000 1.5 ck"
# ============================================
with open('./data/labels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for r in csv_reader:
        row = r[0].split()  # separa por espacio
        rows.append((row[0], row[1], row[2]))  # (video_id, tiempo/end, etiquetas)
        line_count += 1
    print(f'Processed {line_count} lines.')

# ============================================
# CREAR ARCHIVO output_vectors.csv
# Cada línea representa un frame con etiquetas binarias
# ============================================
file = open("./data/output_vectors.csv", "w")
frame_number = 0  # contador global de frames

for row in rows:
    video_id, time_or_end, label_str = row

    # Si es la última sección del video (etiquetada como "end")
    if "end" in time_or_end:
        end_frame = int(get_total_frames(video_id))
    else:
        end_frame = int(float(time_or_end) * fps)  # tiempo → cantidad de frames

    # Generar una línea por cada frame en el rango correspondiente
    for i in range(frame_number, end_frame):
        # Convertir etiquetas (c, k, h, r, x, i) en vectores binarios
        c = 1 if "c" in label_str else 0
        k = 1 if "k" in label_str else 0
        h = 1 if "h" in label_str else 0
        r = 1 if "r" in label_str else 0
        x = 1 if "x" in label_str else 0
        i_label = 1 if "i" in label_str else 0  # Se llama "i" pero se renombra para no chocar con el índice i

        frame_number += 1

        # Formato: video_id, frame_number, c, k, h, r, x, i
        line = "{},{},{},{},{},{},{},{}\n".format(
            video_id, frame_number, c, k, h, r, x, i_label
        )
        file.write(line)

    file.flush()

    # Reiniciar contador si se llega al final del video
    if "end" in time_or_end:
        frame_number = 0

# Cerrar archivo CSV al finalizar
file.close()
