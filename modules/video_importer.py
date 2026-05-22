import cv2
import os
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from modules.hand_detector import HandDetector
from modules.data_recorder import DataRecorder

class VideoImporterThread(QThread):
    # Señales para comunicación fluida con la UI sin congelamiento
    progress_changed = pyqtSignal(int)
    status_msg = pyqtSignal(str)
    frame_processed = pyqtSignal(np.ndarray, str) # frame anotado, mano detectada/info
    finished_successfully = pyqtSignal(int) # total de frames grabados con manos

    def __init__(self, video_path, label, skip_frames=1, min_confidence=0.4):
        super().__init__()
        self.video_path = video_path
        self.label = label
        self.skip_frames = skip_frames
        self.min_confidence = min_confidence
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        if not os.path.exists(self.video_path):
            self.status_msg.emit("Error: El archivo de video no existe.")
            self.finished_successfully.emit(0)
            return

        self.status_msg.emit(f"Iniciando procesamiento de {os.path.basename(self.video_path)}...")
        
        # Inicializar detector configurando min_tracking_confidence=0.4 para tolerar motion blur
        detector = HandDetector(
            max_num_hands=2, 
            min_detection_confidence=self.min_confidence, 
            min_tracking_confidence=self.min_confidence
        )
        recorder = DataRecorder()

        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            self.status_msg.emit("Error: No se pudo leer la duración del video.")
            cap.release()
            self.finished_successfully.emit(0)
            return

        recorded_count = 0
        frame_idx = 0

        while cap.isOpened() and self._is_running:
            ret, frame = cap.read()
            if not ret:
                break

            # Omitir frames según skip_frames
            if frame_idx % (self.skip_frames + 1) == 0:
                # Procesar mano. detect(frame, is_video=True) ya que proviene de un MP4 importado
                vector, annotated_frame = detector.detect(frame, is_video=True)
                
                # Comprobar si hay al menos una mano detectada (el vector no debe ser completamente nulo)
                # Si ambas manos son solo ceros, la suma absoluta es cero (no hay manos)
                if np.any(vector):
                    recorder.record(self.label, vector)
                    recorded_count += 1
                    hand_info = "Manos Detectadas"
                else:
                    hand_info = "Sin Manos"

                # Emitir frame para preview en UI
                self.frame_processed.emit(annotated_frame, hand_info)

            # Notificar progreso
            progress = int((frame_idx / total_frames) * 100)
            self.progress_changed.emit(progress)
            
            frame_idx += 1

        cap.release()
        self.status_msg.emit(f"Procesamiento finalizado. Se grabaron {recorded_count} muestras en el dataset.")
        self.finished_successfully.emit(recorded_count)
