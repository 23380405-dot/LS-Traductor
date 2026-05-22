import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def _normalize_hand(self, landmarks, mirror=False):
        """
        Normaliza los 21 puntos landmarks con invarianza de traslación, escala y rotación.
        El punto 0 se toma como origen y se calcula la rotación alineando el vector 0->9 al eje vertical.
        """
        coords = np.array([[lm.x, lm.y] for lm in landmarks])
        wrist = coords[0]
        coords_rel = coords - wrist
        
        # Invarianza de Escala
        ref_vec = coords_rel[9]
        dist = np.linalg.norm(ref_vec)
        if dist < 1e-5:
            # Fallback si la distancia al punto 9 es nula
            max_dist = 0
            for rx, ry in coords_rel:
                d = np.sqrt(rx*rx + ry*ry)
                if d > max_dist:
                    max_dist = d
            dist = max_dist if max_dist > 1e-5 else 1.0
            
        ref_x, ref_y = ref_vec[0], ref_vec[1]
        ux = ref_x / dist
        uy = ref_y / dist
        
        coords_scaled = []
        for rx, ry in coords_rel:
            # Proyección rotacional para pura invarianza de rotación (coincide con frontend)
            rx_rot = rx * uy - ry * ux
            ry_rot = rx * ux + ry * uy
            
            scaled_x = rx_rot / dist
            scaled_y = ry_rot / dist
            
            # Espejar coordenada X si es la mano izquierda para invarianza de handedness
            if mirror:
                scaled_x = -scaled_x
                
            coords_scaled.append([scaled_x, scaled_y])
            
        return np.array(coords_scaled).flatten()

    def detect(self, frame, is_video=False):
        """
        Detecta manos, dibuja landmarks y retorna un vector plano de 84 dimensiones:
        [42 valores mano izquierda, 42 valores mano derecha]
        Si falta alguna mano, se aplica Zero-Padding, o bien se duplica si solo hay una.
        Aplica un Fix de Efecto Espejo si is_video=False.
        """
        # Convertir a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Inicializar vectores de 42 elementos en cero
        left_hand_vector = np.zeros(42)
        right_hand_vector = np.zeros(42)
        
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks and results.multi_handedness:
            # Dibujar landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    annotated_frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
            
            # 1. Procesar cada mano detectada
            for index, hand_handedness in enumerate(results.multi_handedness):
                label = hand_handedness.classification[0].label # 'Left' o 'Right'
                landmarks = results.multi_hand_landmarks[index].landmark
                
                # Fix de efecto espejo para cámara en vivo (cv2.flip)
                if not is_video:
                    if label == 'Left':
                        label = 'Right'
                    else:
                        label = 'Left'
                
                # Normalizar con espejo si es mano izquierda para que sea comparable a mano derecha
                normalized_vector = self._normalize_hand(landmarks, mirror=(label == 'Left'))
                
                if label == 'Left':
                    left_hand_vector = normalized_vector
                else:
                    right_hand_vector = normalized_vector
            
            # 2. Si sólo hay una mano, colocarla en ambos vectores para que las señas simples sean independientes de la mano usada
            if len(results.multi_hand_landmarks) == 1:
                single_label = results.multi_handedness[0].classification[0].label
                if not is_video:
                    single_label = 'Right' if single_label == 'Left' else 'Left'
                
                single_norm = self._normalize_hand(results.multi_hand_landmarks[0].landmark, mirror=(single_label == 'Left'))
                left_hand_vector = single_norm
                right_hand_vector = single_norm
                    
        # Concatenar para obtener el vector definitivo de 84 dimensiones
        final_vector = np.concatenate([left_hand_vector, right_hand_vector])
        return final_vector, annotated_frame
