import os
import cv2
import pickle
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QTimer
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QLineEdit, QComboBox, QProgressBar,
    QFileDialog, QMessageBox, QDialog, QFormLayout, QGroupBox,
    QSpinBox, QDoubleSpinBox
)
from PyQt6.QtGui import QImage, QPixmap, QFont
from modules.hand_detector import HandDetector
from modules.data_recorder import DataRecorder
from modules.video_importer import VideoImporterThread
from modules.model_trainer import ModelTrainerThread
from modules.tts_engine import TTSEngine

# Constantes de Estilo QSS Moderno (Material / Fluent Dark)
QSS_STYLE = """
QMainWindow {
    background-color: #121214;
}
QWidget {
    background-color: #121214;
    color: #E2E8F0;
    font-family: 'Segoe UI', -apple-system, Roboto, Helvetica, sans-serif;
    font-size: 13px;
}
QDialog {
    background-color: #1A1A1E;
    border: 1px solid #2D3748;
    border-radius: 8px;
}
QTabWidget::pane {
    border: 1px solid #2D3748;
    background-color: #1A1A1E;
    border-radius: 8px;
}
QTabBar::tab {
    background-color: #121214;
    color: #A0AEC0;
    padding: 10px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    border: 1px solid #2D3748;
    border-bottom: none;
    margin-right: 4px;
}
QTabBar::tab:selected, QTabBar::tab:hover {
    background-color: #1A1A1E;
    color: #3182CE;
    font-weight: bold;
    border-top: 2px solid #3182CE;
}
QGroupBox {
    border: 1px solid #2D3748;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: #4A5568;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: #3182CE;
}
QPushButton {
    background-color: #3182CE;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #2B6CB0;
}
QPushButton:pressed {
    background-color: #2C5282;
}
QPushButton:disabled {
    background-color: #2D3748;
    color: #718096;
}
QPushButton#btn_stop_record {
    background-color: #E53E3E;
}
QPushButton#btn_stop_record:hover {
    background-color: #C53030;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #2D3748;
    color: #F7FAFC;
    border: 1px solid #4A5568;
    padding: 6px 12px;
    border-radius: 6px;
}
QLineEdit:focus, QComboBox:focus {
    border: 1px solid #3182CE;
}
QProgressBar {
    border: 1px solid #2D3748;
    border-radius: 4px;
    text-align: center;
    background-color: #1A1A1E;
    color: white;
}
QProgressBar::chunk {
    background-color: #3182CE;
    width: 20px;
}
QLabel#lbl_detección {
    font-size: 32px;
    font-weight: bold;
    color: #48BB78;
    background-color: #1A1A1E;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #2D3748;
}
QLabel#lbl_quiosco_sign {
    font-size: 48px;
    font-weight: bold;
    color: #3182CE;
}
"""

class LoginDialog(QDialog):
    """
    Diálogo inicial de selección de rol y autenticación.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Acceso al Sistema - Señas")
        self.setFixedSize(360, 240)
        self.setStyleSheet(QSS_STYLE)
        
        self.selected_role = "Usuario"
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        title = QLabel("Reconocedor de Señas")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #3182CE; margin-bottom: 10px;")
        layout.addWidget(title)
        
        form_layout = QFormLayout()
        
        self.combo_role = QComboBox()
        self.combo_role.addItems(["Usuario", "Administrador"])
        self.combo_role.currentIndexChanged.connect(self.on_role_changed)
        form_layout.addRow("Seleccionar Rol:", self.combo_role)
        
        self.txt_password = QLineEdit()
        self.txt_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.txt_password.setPlaceholderText("Clave de Administrador")
        self.txt_password.setEnabled(False) # Desactivado por defecto para 'Usuario'
        form_layout.addRow("Contraseña:", self.txt_password)
        
        layout.addLayout(form_layout)
        layout.addSpacing(15)
        
        self.btn_login = QPushButton("Ingresar")
        self.btn_login.clicked.connect(self.validate_access)
        layout.addWidget(self.btn_login)

    def on_role_changed(self, index):
        is_admin = (self.combo_role.currentText() == "Administrador")
        self.txt_password.setEnabled(is_admin)
        if not is_admin:
            self.txt_password.clear()

    def validate_access(self):
        role = self.combo_role.currentText()
        if role == "Administrador":
            password = self.txt_password.text()
            if password == "admin123":
                self.selected_role = "Administrador"
                self.accept()
            else:
                QMessageBox.warning(self, "Acceso Denegado", "La contraseña ingresada es incorrecta.")
        else:
            self.selected_role = "Usuario"
            self.accept()


class CameraThread(QThread):
    """
    Hilo de ejecución dedicado que captura la cámara web, ejecuta la detección en MediaPipe
    y emite señales con el frame procesado y el vector resultante para evitar el frizado del hilo principal.
    """
    frame_ready = pyqtSignal(np.ndarray, np.ndarray) # frame, vector_84

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self._is_running = True
        self.detector = HandDetector(max_num_hands=2)

    def stop(self):
        self._is_running = False

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        
        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Invertir el frame horizontalmente (efecto espejo natural)
            frame_mirrored = cv2.flip(frame, 1)
            
            # Procesar el frame usando el corrector de efecto espejo (is_video=False porque es webcam)
            vector, annotated_frame = self.detector.detect(frame_mirrored, is_video=False)
            
            self.frame_ready.emit(annotated_frame, vector)
            self.msleep(30) # ~30 FPS
            
        cap.release()


class MainWindow(QMainWindow):
    def __init__(self, role="Usuario"):
        super().__init__()
        self.role = role
        self.setWindowTitle("Reconocimiento de Lengua de Señas en Tiempo Real")
        self.setMinimumSize(1000, 700)
        self.setStyleSheet(QSS_STYLE)
        
        # Inicializar Registrador de datos y motor de síntesis de voz
        self.recorder = DataRecorder()
        self.tts = TTSEngine()
        self.tts.start()
        
        # Buffer de suavizado de predicción: Promedio Móvil de probabilidades (Ventana=12)
        self.confidence_window = []
        self.window_size = 12
        self.consecutive_no_hands = 0
        
        # Modelo cargado de clasificación de señas
        self.model = None
        self.load_model()
        
        # Hilos activos
        self.camera_thread = None
        self.importer_thread = None
        self.trainer_thread = None
        
        # Estado de grabación manual en vivo
        self.is_recording_samples = False
        self.current_recording_label = ""
        
        self.init_ui()
        self.start_camera()

    def load_model(self):
        model_path = "sign_model.pkl"
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print("Modelo cargado con éxito.")
            except Exception as e:
                print(f"Error cargando el modelo: {e}")

    def init_ui(self):
        # Crear widget central y layout raíz
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Contenedor Izquierdo: Visualización de Cámara Principal (Común para ambos modos)
        camera_panel = QVBoxLayout()
        
        self.lbl_camera = QLabel("Abriendo Cámara...")
        self.lbl_camera.setMinimumSize(640, 480)
        self.lbl_camera.setStyleSheet("border: 2px solid #2D3748; background-color: #0F172A; border-radius: 10px;")
        self.lbl_camera.setAlignment(Qt.AlignmentFlag.AlignCenter)
        camera_panel.addWidget(self.lbl_camera)
        
        # Panel de Resultados de Predicciones
        self.results_group = QGroupBox("Resultado de la Predicción")
        results_layout = QVBoxLayout(self.results_group)
        
        self.lbl_prediction = QLabel("Esperando manos...")
        self.lbl_prediction.setObjectName("lbl_detección")
        self.lbl_prediction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.lbl_prediction)
        
        # Barra de confianza suavizada
        self.progress_confidence = QProgressBar()
        self.progress_confidence.setValue(0)
        self.progress_confidence.setFormat("Confianza: %p%")
        results_layout.addWidget(self.progress_confidence)
        
        camera_panel.addWidget(self.results_group)
        main_layout.addLayout(camera_panel, stretch=3)
        
        # Contenedor Derecho: Configuración interactiva o Modo Quiosco según Rol
        if self.role == "Usuario":
            # Modo Quiosco Inmersivo: Ocultar los paneles de control de pestañas
            quiosco_panel = QVBoxLayout()
            quiosco_panel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            lbl_info = QLabel("MODO QUIOSCO ACTIVO")
            lbl_info.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            lbl_info.setStyleSheet("color: #4A5568;")
            lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            quiosco_panel.addWidget(lbl_info)
            
            ico_quiosco = QLabel("👐")
            ico_quiosco.setFont(QFont("Segoe UI", 80))
            ico_quiosco.setAlignment(Qt.AlignmentFlag.AlignCenter)
            quiosco_panel.addWidget(ico_quiosco)
            
            self.lbl_quiosco_title = QLabel("Seña:")
            self.lbl_quiosco_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            quiosco_panel.addWidget(self.lbl_quiosco_title)
            
            self.lbl_quiosco_sign = QLabel("-")
            self.lbl_quiosco_sign.setObjectName("lbl_quiosco_sign")
            self.lbl_quiosco_sign.setAlignment(Qt.AlignmentFlag.AlignCenter)
            quiosco_panel.addWidget(self.lbl_quiosco_sign)
            
            quiosco_container = QGroupBox("Modo Inmersivo de Traducción de Señas")
            quiosco_container.setLayout(quiosco_panel)
            main_layout.addWidget(quiosco_container, stretch=1)
        else:
            # Modo Administrador Completo con Pestañas
            self.tabs = QTabWidget()
            
            # Pestaña 1: Grabar en Vivo
            tab_record = QWidget()
            self.setup_record_tab(tab_record)
            self.tabs.addTab(tab_record, "Grabar Dataset")
            
            # Pestaña 2: Importar Video MP4
            tab_import = QWidget()
            self.setup_import_tab(tab_import)
            self.tabs.addTab(tab_import, "Importar MP4")
            
            # Pestaña 3: Entrenar Modelo
            tab_train = QWidget()
            self.setup_train_tab(tab_train)
            self.tabs.addTab(tab_train, "Entrenar Inteligencia")
            
            main_layout.addWidget(self.tabs, stretch=2)

    def setup_record_tab(self, tab):
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        
        lbl_inst = QLabel("Grabe vectores de 84 puntos directamente desde su cámara web:")
        lbl_inst.setWordWrap(True)
        layout.addWidget(lbl_inst)
        
        form = QFormLayout()
        self.txt_record_label = QLineEdit()
        self.txt_record_label.setPlaceholderText("Ej. HOLA, CON GUSTO, GRACIAS")
        form.addRow("Etiqueta o Gesto:", self.txt_record_label)
        layout.addLayout(form)
        
        # Botones de Grabación
        self.btn_start_record = QPushButton("Activar Grabación (Continuo)")
        self.btn_start_record.clicked.connect(self.start_sample_recording)
        layout.addWidget(self.btn_start_record)
        
        self.btn_stop_record = QPushButton("Detener Grabación")
        self.btn_stop_record.setObjectName("btn_stop_record")
        self.btn_stop_record.setEnabled(False)
        self.btn_stop_record.clicked.connect(self.stop_sample_recording)
        layout.addWidget(self.btn_stop_record)
        
        # Visualizador de Estadísticas en vivo
        self.stats_group = QGroupBox("Estadísticas del Dataset")
        self.stats_layout = QVBoxLayout(self.stats_group)
        self.lbl_stats = QLabel("No hay muestras grabadas aún.")
        self.stats_layout.addWidget(self.lbl_stats)
        layout.addWidget(self.stats_group)
        self.refresh_stats()
        
        layout.addStretch()

    def setup_import_tab(self, tab):
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        
        lbl_inst = QLabel("Cargue un video grabado de señas (MP4) para asimilarlos automáticamente al dataset de entrenamiento.")
        lbl_inst.setWordWrap(True)
        layout.addWidget(lbl_inst)
        
        form = QFormLayout()
        
        self.lbl_selected_video = QLabel("Ningún video seleccionado")
        self.lbl_selected_video.setStyleSheet("color: #A0AEC0;")
        self.btn_select_video = QPushButton("Seleccionar Archivo MP4")
        self.btn_select_video.clicked.connect(self.select_video_file)
        form.addRow(self.btn_select_video, self.lbl_selected_video)
        
        self.txt_import_label = QLineEdit()
        self.txt_import_label.setPlaceholderText("Ej. GRACIAS")
        form.addRow("Asignar Etiqueta:", self.txt_import_label)
        
        self.spin_skip_frames = QSpinBox()
        self.spin_skip_frames.setRange(0, 10)
        self.spin_skip_frames.setValue(1) # Valor por defecto skip_frames=1
        form.addRow("Omitir Frames (Frecuencia):", self.spin_skip_frames)
        
        layout.addLayout(form)
        
        self.btn_run_import = QPushButton("Iniciar Procesamiento de Video")
        self.btn_run_import.clicked.connect(self.run_video_importation)
        layout.addWidget(self.btn_run_import)
        
        self.progress_import = QProgressBar()
        self.progress_import.setValue(0)
        layout.addWidget(self.progress_import)
        
        self.lbl_import_status = QLabel("Estado: En espera")
        self.lbl_import_status.setWordWrap(True)
        layout.addWidget(self.lbl_import_status)
        
        layout.addStretch()

    def setup_train_tab(self, tab):
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        
        lbl_inst = QLabel("Ejecute el entrenamiento de la Inteligencia Artificial usando clasificadores inteligentes tolerantes a fallos.")
        lbl_inst.setWordWrap(True)
        layout.addWidget(lbl_inst)
        
        self.btn_run_training = QPushButton("Entrenar Modelo Clasificador")
        self.btn_run_training.clicked.connect(self.run_model_training)
        layout.addWidget(self.btn_run_training)
        
        self.lbl_train_status = QLabel("Estado: Inactivo")
        self.lbl_train_status.setWordWrap(True)
        layout.addWidget(self.lbl_train_status)
        
        self.progress_train = QProgressBar()
        self.progress_train.setValue(0)
        layout.addWidget(self.progress_train)
        
        layout.addStretch()

    def start_camera(self):
        self.camera_thread = CameraThread(0)
        self.camera_thread.frame_ready.connect(self.process_camera_frame)
        self.camera_thread.start()

    def refresh_stats(self):
        if self.role == "Usuario":
            return
        stats = self.recorder.get_labels_stats()
        if not stats:
            self.lbl_stats.setText("Dataset vacío. Agregue muestras en 'dataset.csv'")
            return
            
        text = "Muestras por etiqueta:\n"
        for label, count in stats.items():
            text += f"• <b>{label}</b>: {count} registros\n"
        self.lbl_stats.setText(text)

    # ACCIONES DE GRABACIÓN DE SEÑAS EN VIVO
    def start_sample_recording(self):
        label = self.txt_record_label.text().strip().upper()
        if not label:
            QMessageBox.warning(self, "Campos Vacíos", "Ingrese una etiqueta válida antes de grabar.")
            return
            
        self.current_recording_label = label
        self.is_recording_samples = True
        self.btn_start_record.setEnabled(False)
        self.btn_stop_record.setEnabled(True)
        self.txt_record_label.setEnabled(False)

    def stop_sample_recording(self):
        self.is_recording_samples = False
        self.btn_start_record.setEnabled(True)
        self.btn_stop_record.setEnabled(False)
        self.txt_record_label.setEnabled(True)
        self.refresh_stats()
        QMessageBox.information(self, "Grabación Exitosa", f"Se han registrado muestras para la seña: {self.current_recording_label}")

    # ACCIONES DE IMPORTACIÓN DE VIDEO
    def select_video_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Seleccionar Video de Señas", "", "Videos (*.mp4 *.avi *.mov)")
        if file:
            self.lbl_selected_video.setText(file)

    def run_video_importation(self):
        video = self.lbl_selected_video.text()
        if video == "Ningún video seleccionado" or not os.path.exists(video):
            QMessageBox.warning(self, "Falta Archivo", "Por favor, seleccione un video de entrada válido.")
            return
            
        label = self.txt_import_label.text().strip().upper()
        if not label:
            QMessageBox.warning(self, "Etiqueta faltante", "Por favor ingresa qué seña representa este video.")
            return

        self.btn_run_import.setEnabled(False)
        self.progress_import.setValue(0)
        
        # QThread dedicado de importador de video
        self.importer_thread = VideoImporterThread(
            video_path=video,
            label=label,
            skip_frames=self.spin_skip_frames.value(),
            min_confidence=0.4
        )
        self.importer_thread.progress_changed.connect(self.progress_import.setValue)
        self.importer_thread.status_msg.connect(self.lbl_import_status.setText)
        self.importer_thread.finished_successfully.connect(self.on_import_finished)
        self.importer_thread.start()

    def on_import_finished(self, count):
        self.btn_run_import.setEnabled(True)
        self.refresh_stats()

    # ACCIONES DE ENTRENAMIENTO DE INTELIGENCIA
    def run_model_training(self):
        self.btn_run_training.setEnabled(False)
        self.progress_train.setValue(0)
        
        # QThread de entrenamiento
        self.trainer_thread = ModelTrainerThread()
        self.trainer_thread.progress_changed.connect(self.progress_train.setValue)
        self.trainer_thread.status_msg.connect(self.lbl_train_status.setText)
        self.trainer_thread.training_finished.connect(self.on_training_finished)
        self.trainer_thread.start()

    def on_training_finished(self, success, msg):
        self.btn_run_training.setEnabled(True)
        if success:
            QMessageBox.information(self, "Entrenamiento Listo", msg)
            self.load_model() # Recargar el modelo recién guardado
        else:
            QMessageBox.critical(self, "Fallo", msg)

    # EVENTO DE PROCESADO DE FRAME DE CÁMARA (Lógica Central del Hilo)
    @pyqtSlot(np.ndarray, np.ndarray)
    def process_camera_frame(self, frame, vector):
        # 1. Grabación si el modo grabación de muestras continuo está activo
        # Se guarda el vector en el dataset bajo la etiqueta elegida.
        if self.is_recording_samples and np.any(vector):
            self.recorder.record(self.current_recording_label, vector)

        # 2. Inferencia y predicción en Tiempo Real usando el clasificador entrenado
        has_hands = np.any(vector)
        
        if not has_hands:
            # LÓGICA 'No Seña': Si no hay manos por 10 frames consecutivos, limpiar pantalla
            self.consecutive_no_hands += 1
            if self.consecutive_no_hands >= 10:
                self.lbl_prediction.setText("Esperando manos...")
                self.progress_confidence.setValue(0)
                if self.role == "Usuario":
                    self.lbl_quiosco_sign.setText("-")
                self.confidence_window.clear()
        else:
            self.consecutive_no_hands = 0
            
            if self.model is not None:
                try:
                    # Preparar para predicción (clasificador entrenado con matrices)
                    features = vector.reshape(1, -1)
                    prediction_label = self.model.predict(features)[0]
                    
                    # Calcular confianza del modelo para suavizado
                    if hasattr(self.model, "predict_proba"):
                        probs = self.model.predict_proba(features)[0]
                        max_idx = np.argmax(probs)
                        confidence = float(probs[max_idx]) * 100
                    else:
                        confidence = 100.0 # Valor predeterminado de aproximación
                        
                    # Filtrado por promedio móvil (Ventana=12) para evitar oscilación rápida
                    self.confidence_window.append(confidence)
                    if len(self.confidence_window) > self.window_size:
                        self.confidence_window.pop(0)
                    
                    smoothed_confidence = sum(self.confidence_window) / len(self.confidence_window)
                    
                    # Actualizar UI
                    result_text = f"Predicción: {prediction_label}"
                    self.lbl_prediction.setText(result_text)
                    self.progress_confidence.setValue(int(smoothed_confidence))
                    
                    if self.role == "Usuario":
                        self.lbl_quiosco_sign.setText(prediction_label)
                        
                    # Reproducir por voz usando debounce inteligente en el motor TTS
                    if smoothed_confidence > 75.0: # Sólo hablar si supera el 75% de confianza suavizada
                        self.tts.speak(prediction_label)
                        
                except Exception as e:
                    self.lbl_prediction.setText(f"Inferencia error: {str(e)[:25]}")
            else:
                self.lbl_prediction.setText("Manos detectadas (Modelo sin entrenar)")
                self.progress_confidence.setValue(0)

        # 3. Renderizar el frame a QPixmap para pintarlo en el widget QLabel de PyQt6 sin parpadeos
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Ajustar frame de la cámara para que resulte fluido
        self.lbl_camera.setPixmap(pixmap.scaled(
            self.lbl_camera.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))

    def closeEvent(self, event):
        """Asegurar de cerrar y detener todos los hilos concurrentemente al salir"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
        if self.importer_thread:
            self.importer_thread.stop()
            self.importer_thread.wait()
        if self.tts:
            self.tts.stop()
        event.accept()
