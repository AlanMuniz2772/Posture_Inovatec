# Importación de librerías necesarias
import csv             # Para leer archivos CSV
import numpy as np     # Para trabajar con arreglos numéricos

# Función que carga y procesa los datos desde archivos CSV
def get_data():
    # Abrir archivos CSV de entrada (features) y salida (labels)
    output_vectors = open('./data/output_vectors.csv')
    input_vectors = open('./data/input_vectors.csv')

    # Crear lectores de CSV
    output_reader = csv.reader(output_vectors, delimiter=',')
    input_reader = csv.reader(input_vectors, delimiter=',')

    # Inicializar contador de líneas
    line_count = 0

    # Listas para almacenar los vectores procesados
    outputs = []
    inputs = []

    # Iterar sobre las filas del archivo de salida
    for outl in output_reader:
        line_count += 1

        # Si la última columna de la fila de salida es 1, se ignora esa línea
        if outl[-1] == '1':  # Ojo: se compara con cadena, como vienen del CSV
            next(input_reader)  # También avanzar en el archivo de entrada
            continue

        # Leer la línea correspondiente del archivo de entrada
        inl = next(input_reader)

        # Imprimir las líneas para depuración (se puede comentar si no se usa)
        print(outl)
        print(inl)

        # Asegurar que las dos primeras columnas coincidan entre ambos archivos
        assert outl[0] == inl[0]
        assert outl[1] == inl[1]

        # Agregar los datos de salida a la lista (posición 2 a 6, convertido a float)
        outputs.append((
            float(outl[2]), float(outl[3]), float(outl[4]),
            float(outl[5]), float(outl[6])
        ))

        # Agregar los datos de entrada con control de errores por si hay caracteres extraños
        try:
            inputs.append((
                float(inl[2]), float(inl[3]), float(inl[4]),
                float(inl[5]), float(inl[6])
            ))
        except ValueError:
            # En caso de que el valor venga como una lista o string raro (como ['1.23']), se extrae el número
            inputs.append((
                float(inl[2][1]), float(inl[3][1]), float(inl[4][1]),
                float(inl[5][1]), float(inl[6][1])
            ))

    # Mostrar la cantidad de líneas procesadas
    print(f'Processed {line_count} lines.')

    # Devolver los vectores como arreglos de NumPy
    return np.array(inputs), np.array(outputs)

# Si el archivo se ejecuta directamente, se imprimen los datos cargados
if __name__ == "__main__":
    print(get_data())
