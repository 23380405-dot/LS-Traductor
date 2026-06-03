import pickle
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from .hand_detector import HandDetector   # para leer FEATURE_SIZE


# ══════════════════════════════════════════════════════════════════════
# QTHREAD DE ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════

class ModelTrainerThread(QThread):
    """
    QThread puro para ejecutar el entrenamiento de scikit-learn sin
    bloquear la interfaz gráfica.

    Al ser un QThread (no un threading.Thread disfrazado de QObject),
    las señales se entregan correctamente al hilo principal de la UI.

    Señales:
        log_message  (str)  → mensajes de progreso paso a paso.
        finished_ok  (dict) → métricas al terminar exitosamente.
        finished_err (str)  → mensaje de error si falla.
    """

    log_message  = pyqtSignal(str)
    finished_ok  = pyqtSignal(dict)
    finished_err = pyqtSignal(str)

    def __init__(
        self,
        trainer:      "ModelTrainer",
        X:            np.ndarray,
        y:            np.ndarray,
        algorithm:    str,
        n_estimators: int,
        parent        = None,
    ) -> None:
        super().__init__(parent)
        self.trainer      = trainer
        self.X            = X
        self.y            = y
        self.algorithm    = algorithm
        self.n_estimators = n_estimators

    def run(self) -> None:
        """Ejecuta el entrenamiento y emite los resultados."""
        try:
            metrics = self.trainer.train(
                self.X,
                self.y,
                algorithm    = self.algorithm,
                n_estimators = self.n_estimators,
                on_progress  = lambda msg: self.log_message.emit(msg),
            )
            self.finished_ok.emit(metrics)

        except Exception as exc:
            self.finished_err.emit(str(exc))


# ══════════════════════════════════════════════════════════════════════
# TRAINER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

class ModelTrainer:
    """
    Gestiona el clasificador de señas: entrenamiento, predicción y
    persistencia en disco.

    Thread-safety:
        El atributo _model está protegido por un RLock para que el
        hilo de cámara (predicción) y el QThread de entrenamiento
        no interfieran entre sí.

    Uso típico:
        trainer = ModelTrainer()
        trainer.load_model()              # cargar sesión anterior

        thread = ModelTrainerThread(trainer, X, y, "random_forest", 150)
        thread.finished_ok.connect(...)
        thread.start()

        # Durante / después del entrenamiento:
        label, conf = trainer.predict(features)   # hilo-safe
    """

    MODEL_FILE   = "sign_model.pkl"
    ENCODER_FILE = "label_encoder.pkl"
    FEATURE_SIZE = HandDetector.FEATURE_SIZE   # 84

    # Umbrales mínimos de datos para entrenar
    MIN_SAMPLES_TOTAL = 10
    MIN_PER_CLASS     = 5
    MIN_CLASSES       = 2

    def __init__(
        self,
        models_dir:           str   = "models",
        confidence_threshold: float = 0.50,
    ) -> None:
        """
        Args:
            models_dir:           Directorio donde guardar los artefactos.
            confidence_threshold: Confianza mínima para reportar predicción.
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self._model_path   = self.models_dir / self.MODEL_FILE
        self._encoder_path = self.models_dir / self.ENCODER_FILE

        self._model:     Optional[object] = None
        self._encoder:   LabelEncoder     = LabelEncoder()
        self._is_trained: bool            = False
        self.confidence_threshold         = confidence_threshold

        # RLock → el mismo hilo puede adquirirlo múltiples veces (reentrant)
        self._lock = threading.RLock()

    # ──────────────────────────────────────────────────────────────────
    # ENTRENAMIENTO (llamado desde ModelTrainThread.run())
    # ──────────────────────────────────────────────────────────────────

    def train(
        self,
        X:            np.ndarray,
        y:            np.ndarray,
        algorithm:    str                          = "random_forest",
        n_estimators: int                          = 150,
        on_progress:  Optional[Callable[[str], None]] = None,
    ) -> Dict:
        """
        Entrena el clasificador con el dataset proporcionado.

        Args:
            X:            Array (n_samples, 84) de features normalizadas.
            y:            Array (n_samples,)    de etiquetas string.
            algorithm:    'random_forest' o 'svm'.
            n_estimators: Número de árboles (solo Random Forest).
            on_progress:  Función opcional para mensajes de progreso.

        Returns:
            dict con: accuracy, classes, n_train, n_test, algorithm, report.

        Raises:
            ValueError: Si los datos no cumplen los requisitos mínimos.
        """
        def _log(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        # ── Validación ─────────────────────────────────────────────────
        _log("Validando datos de entrenamiento…")
        self._validate_data(X, y)

        # ── Codificación ───────────────────────────────────────────────
        _log("Codificando etiquetas…")
        with self._lock:
            y_enc   = self._encoder.fit_transform(y)
            classes = list(self._encoder.classes_)

        # ── División train / test ──────────────────────────────────────
        test_size = 0.20 if len(X) >= 60 else 0.15
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_enc,
            test_size    = test_size,
            random_state = 42,
            stratify     = y_enc,
        )
        _log(f"División: {len(X_tr)} entrenamiento / {len(X_te)} prueba.")

        # ── Construcción del clasificador ──────────────────────────────
        _log(f"Construyendo {algorithm}…")
        if algorithm == "svm":
            clf = SVC(
                kernel                  = "rbf",
                C                       = 10.0,
                gamma                   = "scale",
                probability             = True,
                random_state            = 42,
                decision_function_shape = "ovr",
            )
        else:   # random_forest
            clf = RandomForestClassifier(
                n_estimators     = n_estimators,
                max_depth        = None,
                min_samples_leaf = 1,
                class_weight     = "balanced",
                random_state     = 42,
                n_jobs           = -1,
            )

        # ── Entrenamiento ──────────────────────────────────────────────
        _log("Entrenando modelo (puede tardar varios segundos)…")
        clf.fit(X_tr, y_tr)

        # ── Evaluación ─────────────────────────────────────────────────
        _log("Evaluando en conjunto de prueba…")
        y_pred   = clf.predict(X_te)
        accuracy = accuracy_score(y_te, y_pred)
        report   = classification_report(
            y_te, y_pred,
            target_names  = classes,
            zero_division = 0,
        )

        # ── Actualizar modelo en memoria (hilo-safe) ───────────────────
        with self._lock:
            self._model      = clf
            self._is_trained = True

        # ── Persistir en disco ─────────────────────────────────────────
        _log("Guardando modelo en disco…")
        self._save_artifacts()

        metrics = {
            "accuracy":  round(accuracy * 100, 2),
            "classes":   classes,
            "n_train":   len(X_tr),
            "n_test":    len(X_te),
            "algorithm": algorithm,
            "report":    report,
        }
        _log(
            f"✓ Completado — Precisión: {metrics['accuracy']}% "
            f"| Señas: {', '.join(classes)}"
        )
        return metrics

    def invalidate(self) -> None:
        """
        Invalida el modelo en memoria.
        Llamar después de resetear el dataset para que no se sigan
        haciendo predicciones con un modelo que ya no corresponde
        a los datos actuales.
        """
        with self._lock:
            self._model      = None
            self._is_trained = False

    # ──────────────────────────────────────────────────────────────────
    # PREDICCIÓN (llamado desde CameraThread a ~30 FPS)
    # ──────────────────────────────────────────────────────────────────

    def predict(
        self, features: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Predice la seña a partir del vector de características.

        Args:
            features: Array (84,) float32.

        Returns:
            Tuple(etiqueta, confianza):
              - etiqueta: nombre de la seña o None si confianza < umbral.
              - confianza: probabilidad máxima [0, 1].
        """
        with self._lock:
            if not self._is_trained or self._model is None:
                return None, 0.0

            # Validar dimensionalidad en tiempo de ejecución
            if features.shape[0] != self.FEATURE_SIZE:
                return None, 0.0

            try:
                feat_2d  = features.reshape(1, -1)
                pred_enc = self._model.predict(feat_2d)[0]
                probs    = self._model.predict_proba(feat_2d)[0]
                conf     = float(np.max(probs))
            except Exception:
                return None, 0.0

            if conf < self.confidence_threshold:
                return None, conf

            label = self._encoder.inverse_transform([pred_enc])[0]
            return label, conf

    # ──────────────────────────────────────────────────────────────────
    # PERSISTENCIA
    # ──────────────────────────────────────────────────────────────────

    def _save_artifacts(self) -> None:
        with self._lock:
            with open(self._model_path, "wb") as f:
                pickle.dump(self._model, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self._encoder_path, "wb") as f:
                pickle.dump(self._encoder, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self) -> bool:
        """
        Carga modelo y encoder previamente guardados desde disco.

        Returns:
            True si la carga fue exitosa, False en caso contrario.
        """
        if not (self._model_path.exists() and self._encoder_path.exists()):
            return False

        try:
            with open(self._model_path, "rb") as f:
                model = pickle.load(f)
            with open(self._encoder_path, "rb") as f:
                encoder = pickle.load(f)

            with self._lock:
                self._model      = model
                self._encoder    = encoder
                self._is_trained = True

            return True

        except Exception as exc:
            print(f"[ModelTrainer] Error al cargar modelo: {exc}")
            return False

    # ──────────────────────────────────────────────────────────────────
    # PROPIEDADES
    # ──────────────────────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def get_classes(self) -> List[str]:
        with self._lock:
            return list(self._encoder.classes_) if self._is_trained else []

    # ──────────────────────────────────────────────────────────────────
    # VALIDACIÓN INTERNA
    # ──────────────────────────────────────────────────────────────────

    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Verifica que los datos sean suficientes y dimensionalmente correctos.

        Raises:
            ValueError con mensaje descriptivo si alguna condición falla.
        """
        if X is None or y is None or len(X) == 0:
            raise ValueError("No hay datos de entrenamiento disponibles.")

        # Verificar dimensionalidad del vector de features
        if X.ndim != 2 or X.shape[1] != self.FEATURE_SIZE:
            raise ValueError(
                f"Dimensión incorrecta del dataset. "
                f"Se esperan {self.FEATURE_SIZE} features por muestra, "
                f"pero el dataset tiene {X.shape[1] if X.ndim == 2 else '?'}. "
                f"¿El dataset fue generado con una versión anterior? "
                f"Intenta resetear y regrabar."
            )

        if len(X) < self.MIN_SAMPLES_TOTAL:
            raise ValueError(
                f"Mínimo {self.MIN_SAMPLES_TOTAL} muestras requeridas. "
                f"Actuales: {len(X)}."
            )

        labels, counts = np.unique(y, return_counts=True)

        if len(labels) < self.MIN_CLASSES:
            raise ValueError(
                f"Se necesitan al menos {self.MIN_CLASSES} señas distintas. "
                f"Solo hay: {', '.join(labels)}."
            )

        insuficientes = [
            f"{lab} ({cnt})"
            for lab, cnt in zip(labels, counts)
            if cnt < self.MIN_PER_CLASS
        ]
        if insuficientes:
            raise ValueError(
                f"Las siguientes señas tienen menos de {self.MIN_PER_CLASS} "
                f"muestras: {', '.join(insuficientes)}. "
                f"Graba más ejemplos para ellas."
            )
