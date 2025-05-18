import json
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils as utils
import SquatPosture as sp

# Ruta de entrada y salida
ruta_entrada = "input_vectors_2.jsonl"
ruta_salida = "landmarks_con_errores.jsonl"

totales = 0
with open(ruta_entrada, "r") as fin, open(ruta_salida, "w") as fout:
    for line in fin:
        frame_data = json.loads(line)
        raw_landmarks = frame_data["landmarks"]
        points = {k: np.array(v) if v is not None else None for k, v in raw_landmarks.items()}
        
        resultado = sp.calcular_deadlift(points)
        
        frame_data["errores"] = resultado
        count = 0
        for k, v in resultado.items():
            if v:
                count += 1
            if count == 4:
                frame_data["ejecucion_correcta"] = True
                totales += 1	
        # Convierte los landmarks de nuevo a listas
        frame_data["landmarks"] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else None)
            for k, v in points.items()
        }

        # Convierte los resultados de errores a tipos puros de Python
        frame_data["errores"] = {
            k: (bool(v) if isinstance(v, np.bool_) else v)
            for k, v in resultado.items()
        }

        frame_data["ejecucion_correcta"] = bool(frame_data["ejecucion_correcta"]) if "ejecucion_correcta" in frame_data else False

        fout.write(json.dumps(frame_data) + "\n")

print(totales)