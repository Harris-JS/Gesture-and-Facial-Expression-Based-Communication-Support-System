import threading
import queue

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

class SpeechModule:
    def __init__(self):
        self.engine_initialized = False
        self.queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
    def _worker(self):
        if not pyttsx3:
            return
            
        # Initialize in the worker thread to avoid COM MTA/STA threading issues on Windows
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150) # Speed
            self.engine_initialized = True
        except Exception as e:
            print(f"TTS Init Error: {e}")
            return

        while True:
            text = self.queue.get()
            if text is None:
                break
                
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
            self.queue.task_done()

    def speak(self, text):
        if not pyttsx3 or not text:
            return
            
        # Prevent queue from getting too backed up
        if self.queue.qsize() < 2:
            self.queue.put(text)
            
    def shutdown(self):
        self.queue.put(None)
        self.worker_thread.join(timeout=1.0)
