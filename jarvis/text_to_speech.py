"""Text-to-speech using Kokoro TTS for local voice synthesis."""

import os
import warnings
import numpy as np
import config
from audio_utils import play_audio

# Suppress HuggingFace symlinks warning (fix by enabling Windows Developer Mode)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class TextToSpeech:
    """Generates speech from text using Kokoro TTS."""

    SAMPLE_RATE = 24000

    def __init__(self, voice=config.KOKORO_VOICE, lang_code=config.KOKORO_LANG_CODE, speed=config.KOKORO_SPEED):
        """Initialize Kokoro TTS with the specified voice."""
        self.voice = voice
        self.speed = speed

        from kokoro import KPipeline
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pipeline = KPipeline(lang_code=lang_code, repo_id='hexgrad/Kokoro-82M')
        print(f"Kokoro TTS initialized (voice: {self.voice}).")

    def synthesize(self, text):
        """
        Convert text to speech and return audio data.

        Args:
            text: Text string to synthesize

        Returns:
            Tuple of (audio_data as numpy array, sample_rate)
        """
        if not text:
            return np.array([], dtype=np.float32), self.SAMPLE_RATE

        audio_chunks = []
        for gs, ps, audio in self.pipeline(text, voice=self.voice, speed=self.speed):
            audio_chunks.append(audio)

        if not audio_chunks:
            return np.array([], dtype=np.float32), self.SAMPLE_RATE

        audio_array = np.concatenate(audio_chunks)
        return audio_array, self.SAMPLE_RATE

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

    def speak_streaming(self, text_iterator):
        """
        Speak text as it streams in, buffering by sentence for natural speech.

        Args:
            text_iterator: An iterator/generator that yields text chunks
        """
        buffer = ""
        sentence_endings = {'.', '!', '?'}
        full_response = ""

        print("Jarvis: ", end="", flush=True)

        for chunk in text_iterator:
            buffer += chunk
            full_response += chunk
            print(chunk, end="", flush=True)

            # Check for complete sentences and speak them
            while True:
                # Find the earliest sentence ending
                earliest_idx = -1
                for ending in sentence_endings:
                    idx = buffer.find(ending)
                    if idx != -1:
                        if earliest_idx == -1 or idx < earliest_idx:
                            earliest_idx = idx

                if earliest_idx == -1:
                    break

                # Extract the sentence (include the punctuation)
                sentence = buffer[:earliest_idx + 1].strip()
                buffer = buffer[earliest_idx + 1:]

                if sentence:
                    # Synthesize and play this sentence immediately
                    audio_data, sample_rate = self.synthesize(sentence)
                    if len(audio_data) > 0:
                        play_audio(audio_data, sample_rate)

        # Speak any remaining text that didn't end with sentence punctuation
        remaining = buffer.strip()
        if remaining:
            audio_data, sample_rate = self.synthesize(remaining)
            if len(audio_data) > 0:
                play_audio(audio_data, sample_rate)

        print()  # Newline after streaming output


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
