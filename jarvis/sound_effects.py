"""Sound effects for Jarvis UI feedback."""

import numpy as np
import sounddevice as sd


class SoundEffects:
    """Generate and play sound effects for Jarvis."""

    def __init__(self, sample_rate=44100):
        """Initialize sound effects generator.

        Args:
            sample_rate: Audio sample rate (default 44100 Hz for high quality)
        """
        self.sample_rate = sample_rate

    def _generate_tone(self, frequency, duration, volume=0.3):
        """Generate a sine wave tone.

        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds
            volume: Volume level (0.0 to 1.0)

        Returns:
            Numpy array of audio samples
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        tone = volume * np.sin(2 * np.pi * frequency * t)

        # Apply fade in/out to avoid clicks
        fade_samples = int(self.sample_rate * 0.01)  # 10ms fade
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)

        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out

        return tone.astype(np.float32)

    def _play_audio(self, audio_data):
        """Play audio data through speakers.

        Args:
            audio_data: Numpy array of audio samples
        """
        try:
            # Query the default output device
            default_output = sd.query_devices(kind='output')
            output_device = default_output['index'] if default_output else None

            sd.play(audio_data, samplerate=self.sample_rate, device=output_device)
            sd.wait()
        except Exception as e:
            # Silently fail if sound can't play - don't disrupt the flow
            pass

    def play_listening_start(self):
        """Play high-pitched 'dun dun' sound when Jarvis starts listening."""
        # Two ascending tones: C6 (1046 Hz) -> E6 (1318 Hz)
        tone1 = self._generate_tone(1046, 0.12, volume=0.25)
        gap = np.zeros(int(self.sample_rate * 0.05), dtype=np.float32)  # 50ms gap
        tone2 = self._generate_tone(1318, 0.12, volume=0.25)

        audio = np.concatenate([tone1, gap, tone2])
        self._play_audio(audio)

    def play_listening_stop(self):
        """Play low-pitched 'dun dun' sound when Jarvis stops listening to process."""
        # Two descending tones: E4 (329 Hz) -> C4 (261 Hz)
        tone1 = self._generate_tone(329, 0.12, volume=0.25)
        gap = np.zeros(int(self.sample_rate * 0.05), dtype=np.float32)  # 50ms gap
        tone2 = self._generate_tone(261, 0.12, volume=0.25)

        audio = np.concatenate([tone1, gap, tone2])
        self._play_audio(audio)


if __name__ == "__main__":
    # Test the sound effects
    print("Testing Jarvis sound effects...")

    sfx = SoundEffects()

    print("\nPlaying 'listening start' sound (high-pitched dun dun)...")
    sfx.play_listening_start()

    import time
    time.sleep(0.5)

    print("Playing 'listening stop' sound (low-pitched dun dun)...")
    sfx.play_listening_stop()

    print("\nSound effects test complete!")
