"""Wake word detection using Vosk for 'Hey Jarvis'."""

import os
import json
import zipfile
import urllib.request
import numpy as np
from vosk import Model, KaldiRecognizer
import config
from audio_utils import AudioRecorder


class WakeWordDetector:
    """Detects the wake phrase 'Hey Jarvis' using Vosk."""

    def __init__(self):
        self.model = None
        self.recognizer = None
        self.recorder = None
        self._ensure_model_downloaded()
        self._load_model()

    def _ensure_model_downloaded(self):
        """Download Vosk model if not present."""
        model_path = config.WAKE_WORD_MODEL_DIR / "vosk-model-small-en-us-0.15"

        if not model_path.exists():
            print("Downloading Vosk model for wake word detection...")
            zip_path = config.WAKE_WORD_MODEL_DIR / "model.zip"

            # Download the model
            urllib.request.urlretrieve(config.WAKE_WORD_MODEL_URL, zip_path)

            # Extract the model
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(config.WAKE_WORD_MODEL_DIR)

            # Clean up zip file
            os.remove(zip_path)
            print("Vosk model downloaded successfully.")

    def _load_model(self):
        """Load the Vosk model."""
        model_path = config.WAKE_WORD_MODEL_DIR / "vosk-model-small-en-us-0.15"
        self.model = Model(str(model_path))
        self.recognizer = KaldiRecognizer(self.model, config.SAMPLE_RATE)

    def start_listening(self):
        """Start the audio recorder for wake word detection."""
        self.recorder = AudioRecorder()
        self.recorder.start()

    def stop_listening(self):
        """Stop the audio recorder."""
        if self.recorder:
            self.recorder.stop()
            self.recorder = None

    def check_for_wake_word(self):
        """
        Check if the wake word was detected in the audio stream.
        Returns True if 'Hey Jarvis' was detected.
        """
        if not self.recorder:
            return False

        chunk = self.recorder.get_audio_chunk(timeout=0.5)
        if chunk is None:
            return False

        # Convert float32 to int16 for Vosk
        audio_int16 = (chunk * 32767).astype(np.int16)

        if self.recognizer.AcceptWaveform(audio_int16.tobytes()):
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "").lower()

            if config.WAKE_PHRASE in text:
                print(f"Wake word detected: '{text}'")
                self.recognizer.Reset()
                return True
        else:
            # Check partial results for faster response
            partial = json.loads(self.recognizer.PartialResult())
            partial_text = partial.get("partial", "").lower()

            if config.WAKE_PHRASE in partial_text:
                print(f"Wake word detected (partial): '{partial_text}'")
                self.recognizer.Reset()
                return True

        return False

    def wait_for_wake_word(self):
        """Block until the wake word is detected."""
        print("Listening for 'Hey Jarvis'...")
        self.start_listening()

        try:
            while True:
                if self.check_for_wake_word():
                    # Clear audio buffer to avoid reprocessing
                    self.recorder.clear_queue()
                    return True
        finally:
            self.stop_listening()


if __name__ == "__main__":
    # Test wake word detection
    detector = WakeWordDetector()
    print("Say 'Hey Jarvis' to test wake word detection...")
    detector.wait_for_wake_word()
    print("Wake word detected successfully!")
