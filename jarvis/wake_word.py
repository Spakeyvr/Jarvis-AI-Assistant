"""Wake word detection using Distil-Whisper for 'Hey Jarvis'."""

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import config
from audio_utils import AudioRecorder
from collections import deque


class WakeWordDetector:
    """Detects the wake phrase 'Hey Jarvis' using Distil-Whisper."""

    def __init__(self):
        self.recorder = None
        self.debug = config.WAKE_WORD_DEBUG
        self.audio_buffer = deque(maxlen=6)  # Keep last 6 chunks (~3 seconds) for better context
        self.whisper_pipe = None
        self.sensitivity = config.WAKE_WORD_SENSITIVITY
        self._load_model()

    def _load_model(self):
        """Load the Distil-Whisper model (shared with speech-to-text)."""
        print("Initializing Distil-Whisper for wake word detection...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            config.WHISPER_MODEL,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(config.WHISPER_MODEL)

        self.whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        print("Distil-Whisper ready for wake word detection.")

    def start_listening(self):
        """Start the audio recorder for wake word detection."""
        self.recorder = AudioRecorder()
        self.recorder.start()

    def stop_listening(self):
        """Stop the audio recorder."""
        if self.recorder:
            self.recorder.stop()
            self.recorder = None

    def _check_wake_phrase_match(self, text):
        """Check if text contains wake phrase with sensitivity-based matching."""
        import string

        # Remove all punctuation and convert to lowercase
        text_lower = text.lower().strip()
        text_lower = text_lower.translate(str.maketrans('', '', string.punctuation))

        # Define sensitivity levels
        if self.sensitivity == "high":
            # High sensitivity - catches common variations
            hey_words = ["hey", "hay", "hi"]
            jarvis_words = ["jarvis", "javis", "jarvas", "jarves"]
        elif self.sensitivity == "medium":
            # Balanced - reasonable variations
            hey_words = ["hey", "hay"]
            jarvis_words = ["jarvis", "javis"]
        else:  # low
            # Strict - only close matches
            hey_words = ["hey"]
            jarvis_words = ["jarvis"]

        # Split into words for analysis
        words = text_lower.split()

        # Check for "hey" variations
        has_hey = any(w in hey_words for w in words)

        # Check for "jarvis" variations
        has_jarvis = any(w in jarvis_words for w in words)

        # Must have BOTH hey and jarvis
        if has_hey and has_jarvis:
            return True

        # Also check for phrase matches (in case words are merged)
        for hey in ["hey", "hay", "hi"]:
            for jarvis in jarvis_words:
                phrase = f"{hey} {jarvis}"
                if phrase in text_lower:
                    return True
                # Check for merged words like "heyjarvis"
                merged = f"{hey}{jarvis}"
                if merged in text_lower.replace(" ", ""):
                    return True

        return False

    def check_for_wake_word(self):
        """
        Check if the wake word was detected in the audio stream.
        Returns True if 'Hey Jarvis' was detected.
        """
        if not self.recorder:
            return False

        # Get audio chunk
        chunk = self.recorder.get_audio_chunk(timeout=0.5)
        if chunk is None:
            return False

        # Calculate RMS energy
        rms = np.sqrt(np.mean(chunk**2))

        # Only process if there's sufficient audio energy
        if rms < config.WAKE_WORD_ENERGY_THRESHOLD:
            return False

        # Add to buffer
        self.audio_buffer.append(chunk)

        # Process more frequently (every 2 chunks = ~1 second) for faster response
        if len(self.audio_buffer) >= 2:
            # Concatenate buffer for analysis
            audio_data = np.concatenate(list(self.audio_buffer))

            # Transcribe using Whisper
            try:
                result = self.whisper_pipe(
                    {"raw": audio_data.astype(np.float32), "sampling_rate": config.SAMPLE_RATE},
                    generate_kwargs={"language": "english"},
                    return_timestamps=False
                )

                text = result["text"].strip().lower()

                if self.debug and text:
                    print(f"[DEBUG] Whisper recognized: '{text}' (energy: {rms:.4f}, sensitivity: {self.sensitivity})")

                # Check if wake word is present
                if self._check_wake_phrase_match(text):
                    print(f"Wake word detected: '{text}' (energy: {rms:.4f})")
                    self.audio_buffer.clear()
                    # Clear remaining audio queue to avoid reprocessing
                    self.recorder.clear_queue()
                    return True

            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Whisper error: {e}")

            # Maintain sliding window - remove oldest chunk but keep overlap
            # This ensures we don't miss wake words that span chunk boundaries
            if len(self.audio_buffer) >= 4:
                self.audio_buffer.popleft()

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
