import csv
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils as utils
import SquatPosture as sp

# Archivos de entrada y salida
INPUT_CSV = "./input_vectors_deadlift.csv"
OUTPUT_CSV = "./output_vectors_ddeadlift.csv"

# Abrimos archivos
with open(INPUT_CSV, "r") as fin, open(OUTPUT_CSV, "w", newline="") as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)



    print("📦 Generando etiquetas automáticas en output_vectors.csv...")

    for row in reader:
        video = row[0]
        frame = int(row[1])
        params = list(map(float, row[2:7]))

        # Aplicar función de autoetiquetado
        etiquetas = sp.auto_label(params)

        # Convertir etiquetas a formato binario
        label_flags = {
            "c": 1 if "c" in etiquetas else 0,
            "k": 1 if "k" in etiquetas else 0,
            "h": 1 if "h" in etiquetas else 0,
            "r": 1 if "r" in etiquetas else 0,
            "x": 1 if "x" in etiquetas else 0,
            "i": 1 if "i" in etiquetas else 0,
        }

        writer.writerow([
            video, frame,
            label_flags["c"], label_flags["k"], label_flags["h"],
            label_flags["r"], label_flags["x"], label_flags["i"]
        ])

print("✅ output_vectors.csv generado con etiquetas binarias.")
