"""Text-to-speech using Piper TTS for local voice synthesis."""

import os
import subprocess
import tempfile
import urllib.request
import tarfile
from pathlib import Path
import numpy as np
import soundfile as sf
import config
from audio_utils import play_audio


class TextToSpeech:
    """Generates speech from text using Piper TTS."""

    # Piper voice model URLs
    VOICE_URLS = {
        "en_GB-alan-medium": {
            "model": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
            "config": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json"
        }
    }

    def __init__(self, voice=config.PIPER_VOICE):
        """Initialize Piper TTS with the specified voice."""
        self.voice = voice
        self.voice_dir = config.PIPER_MODEL_DIR
        self.model_path = None
        self.config_path = None
        self._setup_voice()
        self._init_piper()

    def _setup_voice(self):
        """Set up the voice model paths (local Jarvis or downloadable)."""
        # Check if this is a custom Jarvis voice
        if self.voice in config.JARVIS_VOICES:
            voice_dir = config.JARVIS_VOICES[self.voice]
            self.model_path = voice_dir / f"{self.voice}.onnx"
            self.config_path = voice_dir / f"{self.voice}.onnx.json"

            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Jarvis voice model not found: {self.model_path}\n"
                    f"Please ensure the jarvis-voice folder is in the project root."
                )
            print(f"Using custom Jarvis voice: {self.voice}")
        else:
            # Use downloadable voice
            self._ensure_voice_downloaded()

    def _ensure_voice_downloaded(self):
        """Download the Piper voice model if not present."""
        if self.voice not in self.VOICE_URLS:
            raise ValueError(f"Unknown voice: {self.voice}. Available: {list(self.VOICE_URLS.keys()) + list(config.JARVIS_VOICES.keys())}")

        urls = self.VOICE_URLS[self.voice]
        self.model_path = self.voice_dir / f"{self.voice}.onnx"
        self.config_path = self.voice_dir / f"{self.voice}.onnx.json"

        if not self.model_path.exists():
            print(f"Downloading Piper voice model ({self.voice})...")
            urllib.request.urlretrieve(urls["model"], self.model_path)
            print("Voice model downloaded.")

        if not self.config_path.exists():
            print("Downloading voice config...")
            urllib.request.urlretrieve(urls["config"], self.config_path)
            print("Voice config downloaded.")

    def _init_piper(self):
        """Initialize Piper TTS."""
        try:
            from piper import PiperVoice
            self.piper_voice = PiperVoice.load(
                str(self.model_path),
                config_path=str(self.config_path)
            )
            self.use_piper_python = True
            print("Piper TTS initialized (Python library).")
        except ImportError:
            # Fallback to command-line piper if library not available
            self.use_piper_python = False
            print("Piper Python library not found, will use command-line fallback.")

    def synthesize(self, text):
        """
        Convert text to speech and return audio data.

        Args:
            text: Text string to synthesize

        Returns:
            Tuple of (audio_data as numpy array, sample_rate)
        """
        if not text:
            return np.array([]), 22050

        if self.use_piper_python:
            return self._synthesize_python(text)
        else:
            return self._synthesize_cli(text)

    def _synthesize_python(self, text):
        """Synthesize using the Piper Python library."""
        # Collect audio from synthesize generator
        audio_chunks = []
        sample_rate = 22050  # Default, will be updated from chunk

        for chunk in self.piper_voice.synthesize(text):
            # AudioChunk has audio_float_array attribute (numpy float32 array)
            audio_chunks.append(chunk.audio_float_array)
            sample_rate = chunk.sample_rate

        if not audio_chunks:
            return np.array([], dtype=np.float32), sample_rate

        # Combine all chunks
        audio_array = np.concatenate(audio_chunks)

        return audio_array, sample_rate

    def _synthesize_cli(self, text):
        """Synthesize using command-line Piper (fallback)."""
        # Create temp file for output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            # Run piper command
            cmd = [
                "piper",
                "--model", str(self.model_path),
                "--config", str(self.config_path),
                "--output_file", output_path
            ]

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            process.communicate(input=text.encode())

            # Read the generated audio
            audio_data, sample_rate = sf.read(output_path)
            return audio_data.astype(np.float32), sample_rate

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def speak(self, text):
        """
        Convert text to speech and play it.

        Args:
            text: Text string to speak
        """
        if not text:
            return

        print(f"Jarvis: {text}")
        audio_data, sample_rate = self.synthesize(text)

        if len(audio_data) > 0:
            play_audio(audio_data, sample_rate)


if __name__ == "__main__":
    # Test text-to-speech
    tts = TextToSpeech()

    test_texts = [
        "Hello, I am Jarvis, your personal AI assistant.",
        "The weather today is partly cloudy with a high of 72 degrees.",
        "How may I assist you?"
    ]

    for text in test_texts:
        print(f"\nSpeaking: {text}")
        tts.speak(text)
