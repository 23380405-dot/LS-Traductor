import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Optional, Tuple


class HandDetector:
    """
    Detecta hasta 2 manos en un frame BGR y retorna un vector de
    características normalizado de tamaño fijo (FEATURE_SIZE = 84).

    Constantes de clase:
        WRIST_IDX    (0 ): Muñeca — origen de traslación.
        MID_MCP_IDX  (9 ): MCP dedo medio — referencia de escala.
        NUM_LANDMARKS(21): Landmarks por mano.
        HAND_FEATURE (42): Valores por mano (21 × x, y).
        FEATURE_SIZE (84): Vector total = 2 manos × 42.
    """

    WRIST_IDX     = 0
    MID_MCP_IDX   = 9
    NUM_LANDMARKS = 21
    HAND_FEATURE  = 42
    FEATURE_SIZE  = 84

    # Colores BGR para cada mano en el feed visual
    _COLOR_LEFT  = (0, 255, 128)   # verde-cian   → mano izquierda real
    _COLOR_RIGHT = (0, 128, 255)   # azul-naranja  → mano derecha real

    def __init__(
        self,
        max_hands:            int   = 2,
        detection_confidence: float = 0.70,
        tracking_confidence:  float = 0.70,
    ) -> None:
        """
        Args:
            max_hands:             Manos a rastrear simultáneamente (2).
            detection_confidence:  Umbral de detección inicial [0, 1].
            tracking_confidence:   Umbral de seguimiento continuo [0, 1].
        """
        self._mp_hands          = mp.solutions.hands
        self._mp_drawing        = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode        = False,
            max_num_hands            = max_hands,
            min_detection_confidence = detection_confidence,
            min_tracking_confidence  = tracking_confidence,
        )

        self._lm_style   = self._mp_drawing_styles.get_default_hand_landmarks_style()
        self._conn_style = self._mp_drawing_styles.get_default_hand_connections_style()

    # ──────────────────────────────────────────────────────────────────
    # API PÚBLICA
    # ──────────────────────────────────────────────────────────────────

    def detect(
        self,
        frame:    np.ndarray,
        is_video: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict]]:
        """
        Detecta hasta 2 manos en el frame, dibuja landmarks y retorna
        el vector de 84 características normalizado.

        Corrección de espejo integrada:
            is_video=False → INVIERTE la etiqueta de MediaPipe para
                             compensar el cv2.flip previo al llamador.
            is_video=True  → Usa las etiquetas de MediaPipe tal cual.

        Args:
            frame:    Imagen BGR de OpenCV.
            is_video: False = modo cámara (con flip previo);
                      True  = modo importación de .mp4 (sin flip).

        Returns:
            Tuple(frame_anotado, features(84,) | None, hands_info | None).
        """
        rgb               = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results           = self._hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return frame, None, None

        hand_map:  Dict[str, np.ndarray] = {}
        hands_info: Dict[str, float]     = {}

        for hand_lm, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            raw_label = handedness.classification[0].label   # "Left" o "Right"
            score     = handedness.classification[0].score

            # ── CORRECCIÓN DE EFECTO ESPEJO ────────────────────────────
            # is_video=False: el frame viene de cv2.flip() → MediaPipe
            # ve la imagen volteada y reporta la lateralidad invertida.
            # Invertimos la etiqueta ANTES de asignar al slot del vector
            # para que "Left" del vector corresponda a la mano izquierda
            # REAL del usuario, no a la mano izquierda del espejo.
            #
            # is_video=True: el frame viene directo del .mp4 sin voltear,
            # así que las etiquetas de MediaPipe son correctas tal cual.
            if is_video:
                corrected = raw_label
            else:
                corrected = "Right" if raw_label == "Left" else "Left"

            # Dibujar con color según lateralidad CORREGIDA del usuario
            color = self._COLOR_LEFT if corrected == "Left" else self._COLOR_RIGHT
            self._draw_hand(frame, hand_lm, color)

            # Normalizar esta mano → vector (42,)
            hand_map[corrected]   = self._normalize_single_hand(hand_lm)
            hands_info[corrected] = round(score, 3)

        # ── Construir vector de 84 con duplicación si sólo hay una mano ────────────────────
        # Orden fijo: [ mano_izquierda(42) | mano_derecha(42) ]
        zeros     = np.zeros(self.HAND_FEATURE, dtype=np.float32)
        left_vec  = hand_map.get("Left",  None)
        right_vec = hand_map.get("Right", None)

        if left_vec is not None and right_vec is None:
            right_vec = left_vec.copy()
        elif right_vec is not None and left_vec is None:
            left_vec = right_vec.copy()
        elif left_vec is None and right_vec is None:
            left_vec = zeros
            right_vec = zeros

        features  = np.concatenate([left_vec, right_vec])   # → (84,)

        # Dibujar etiquetas IZQ/DER con la corrección aplicada
        h, w = frame.shape[:2]
        self._draw_hand_labels(frame, results, h, w, is_video=is_video)

        return frame, features, hands_info

    def detect_for_video(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Atajo para VideoImportThread: detect con is_video=True."""
        annotated, features, _ = self.detect(frame, is_video=True)
        return annotated, features

    def release(self) -> None:
        """Libera los recursos de MediaPipe."""
        self._hands.close()

    # ──────────────────────────────────────────────────────────────────
    # NORMALIZACIÓN
    # ──────────────────────────────────────────────────────────────────

    def _normalize_single_hand(self, hand_landmarks) -> np.ndarray:
        """
        Convierte los 21 landmarks de UNA mano en vector (42,) normalizado.

        Pasos:
            1. Extraer (x, y) en espacio [0,1] → (21, 2) float32.
            2. Traslación: restar muñeca (índice 0) → origen (0, 0).
            3. Escala: dividir por ‖coords[9]‖ → adimensional.
            4. Aplanar → (42,) float32.
        """
        coords = np.array(
            [[lm.x, lm.y] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )
        coords -= coords[self.WRIST_IDX].copy()
        scale = np.linalg.norm(coords[self.MID_MCP_IDX])
        if scale > 1e-6:
            coords /= scale
        return coords.flatten()

    # ──────────────────────────────────────────────────────────────────
    # DIBUJO
    # ──────────────────────────────────────────────────────────────────

    def _draw_hand(
        self, frame: np.ndarray, hand_lm, color: Tuple[int, int, int]
    ) -> None:
        """Dibuja el esqueleto de una mano con el color dado."""
        lm_spec   = self._mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3)
        conn_spec = self._mp_drawing.DrawingSpec(color=color, thickness=1)
        self._mp_drawing.draw_landmarks(
            image                   = frame,
            landmark_list           = hand_lm,
            connections             = self._mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec   = lm_spec,
            connection_drawing_spec = conn_spec,
        )

    def _draw_hand_labels(
        self,
        frame:    np.ndarray,
        results,
        h: int, w: int,
        is_video: bool = False,
    ) -> None:
        """Dibuja etiquetas 'IZQ'/'DER' sobre cada muñeca, con la misma
        corrección de espejo que se aplica en detect()."""
        if not results.multi_hand_landmarks or not results.multi_handedness:
            return
        for hand_lm, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            raw   = handedness.classification[0].label
            corr  = raw if is_video else ("Right" if raw == "Left" else "Left")
            label = "IZQ" if corr == "Left" else "DER"
            color = self._COLOR_LEFT if corr == "Left" else self._COLOR_RIGHT
            wx    = int(hand_lm.landmark[self.WRIST_IDX].x * w)
            wy    = int(hand_lm.landmark[self.WRIST_IDX].y * h)
            cv2.circle(frame, (wx, wy), 8, (0, 255, 255), 2)
            cv2.putText(
                frame, label, (wx + 10, wy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
