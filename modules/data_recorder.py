import os
import csv
import numpy as np

class DataRecorder:
    def __init__(self, filename="dataset.csv"):
        self.filename = filename

    def record(self, label, vector):
        """
        Guarda un vector de 84 dimensiones asociado a una etiqueta en el dataset CSV.
        Usa Append Inteligente: Si el archivo ya existe, añade la fila al final.
        El formato de cada fila es: label, x1, y1, x2, y2, ..., x42, y42 (para mano izq y der)
        """
        # Crear encabezados si el archivo no existe
        file_exists = os.path.isfile(self.filename)
        
        with open(self.filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Escribir cabecera
                headers = ["label"] + [f"coord_{i}" for i in range(84)]
                writer.writerow(headers)
            
            # Escribir la etiqueta junto al vector de 84 elementos
            row = [label] + list(vector)
            writer.writerow(row)
            
    def get_labels_stats(self):
        """
        Retorna estadísticas rápidas del dataset para la interfaz de administración.
        """
        if not os.path.isfile(self.filename):
            return {}
            
        stats = {}
        with open(self.filename, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                next(reader) # Saltar cabecera
            except StopIteration:
                return {}
                
            for row in reader:
                if row:
                    label = row[0]
                    stats[label] = stats.get(label, 0) + 1
        return stats
