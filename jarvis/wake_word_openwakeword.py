"""Wake word detection using openWakeWord for 'Hey Jarvis'."""

import numpy as np
from openwakeword.model import Model
from openwakeword import utils as oww_utils
import config
from audio_utils import AudioRecorder


class WakeWordDetector:
    """Detects the wake phrase 'Hey Jarvis' using openWakeWord."""

    def __init__(self):
        self.recorder = None
        self.debug = config.WAKE_WORD_DEBUG
        self.model = None
        self.detection_threshold = config.OPENWAKEWORD_THRESHOLD
        self._load_model()

    def _load_model(self):
        """Load the openWakeWord model for 'hey jarvis'."""
        print("Initializing openWakeWord for wake word detection...")

        # Download models if they don't exist
        try:
            print("Downloading/verifying wake word models...")
            oww_utils.download_models()
        except Exception as e:
            print(f"Note: Model download check completed ({e})")

        # Initialize model with 'hey_jarvis' - it will auto-download if needed
        self.model = Model(
            wakeword_models=["hey_jarvis"],
            inference_framework="onnx"
        )

        print(f"openWakeWord ready with model: hey_jarvis (threshold: {self.detection_threshold})")

    def start_listening(self):
        """Start the audio recorder for wake word detection."""
        # Reset the model's internal audio buffer to clear any residual state
        # This prevents false detections from previous audio (e.g., TTS output)
        self.model.reset()

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

        # Get audio chunk (openWakeWord expects 80ms chunks = 1280 samples at 16kHz)
        chunk = self.recorder.get_audio_chunk(timeout=0.5)
        if chunk is None:
            return False

        # Convert to int16 for openWakeWord (expects 16-bit PCM audio)
        audio_data = (chunk * 32767).astype(np.int16)

        # Get predictions from the model
        prediction = self.model.predict(audio_data)

        # Check if 'hey_jarvis' was detected above threshold
        if 'hey_jarvis' in prediction:
            score = prediction['hey_jarvis']

            if self.debug:
                print(f"[DEBUG] Wake word score: {score:.4f}")

            if score >= self.detection_threshold:
                print(f"Wake word detected! (score: {score:.4f})")
                # Clear remaining audio queue to avoid reprocessing
                self.recorder.clear_queue()
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
