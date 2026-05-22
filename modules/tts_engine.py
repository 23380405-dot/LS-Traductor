import time
import queue
import threading

class TTSEngine(threading.Thread):
    """
    Motor de Síntesis de Voz (Text-to-Speech) optimizado para evitar el congelamiento de la UI.
    Ejecuta el ciclo de pyttsx3/SAPI5 en un hilo secundario dedicado aplicando cola de mensajes.
    Contiene un mecanismo de debounce de 2 segundos para evitar repeticiones de la misma palabra.
    """
    def __init__(self, debounce_time=2.0):
        super().__init__()
        self.queue = queue.Queue()
        self.debounce_time = debounce_time
        self.last_spoken = {}
        self._is_running = True
        self.daemon = True # Detener con la app principal

    def speak(self, text):
        """Método público no bloqueante llamado por el hilo principal de la UI."""
        if not text:
            return
            
        now = time.time()
        # Debounce por palabra / seña para no saturar con repeticiones robóticas continuas
        if text in self.last_spoken:
            if now - self.last_spoken[text] < self.debounce_time:
                return # Ignorar repetición dentro del intervalo de debounce
        
        self.last_spoken[text] = now
        self.queue.put(text)

    def stop(self):
        self._is_running = False
        self.queue.put(None) # Desbloquear el loop

    def run(self):
        # NOTA MANDATORIA: pyttsx3.init() DEBE estar dentro del método run() del hilo dedicado
        # para evitar fallos de concurrencia y congelamiento de COM/SAPI5 de Windows.
        try:
            import pyttsx3
            engine = pyttsx3.init()
            
            # Configurar propiedades opcionales
            engine.setProperty('rate', 150) # Velocidad ligeramente pausada
            engine.setProperty('volume', 1.0) # Volumen máximo
            
            # Intentar seleccionar una voz en español
            voices = engine.getProperty('voices')
            for voice in voices:
                if "spanish" in voice.name.lower() or "es-es" in voice.id.lower() or "es-mx" in voice.id.lower():
                    engine.setProperty('voice', voice.id)
                    break
        except Exception as e:
            print(f"Advertencia - No se pudo inicializar pyttsx3: {e}")
            engine = None

        while self._is_running:
            try:
                text = self.queue.get(timeout=0.5)
                if text is None:
                    break
                
                print(f"[TTS Spark] Hablando: {text}")
                
                if engine is not None:
                    engine.say(text)
                    engine.runAndWait()
                    
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en el ciclo del motor TTS: {e}")
