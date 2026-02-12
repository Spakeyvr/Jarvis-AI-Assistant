"""Audio utilities for microphone input and speaker output."""

import numpy as np
import sounddevice as sd
import soundfile as sf
from collections import deque
import threading
import queue
import config


def list_input_devices():
    """List all available input (microphone) devices."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    input_devices = []
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            api_name = hostapis[dev['hostapi']]['name'] if dev['hostapi'] < len(hostapis) else "Unknown"
            input_devices.append((i, dev['name'], dev['max_input_channels'], api_name))
    return input_devices


def print_input_devices():
    """Print available input devices for user selection."""
    print("\nAvailable microphones:")
    print("-" * 60)
    print(f"{'ID':<4} {'Name':<40} {'API':<10}")
    print("-" * 60)
    devices = list_input_devices()

    # Get default input device safely
    default_input = None
    try:
        default_dev = sd.default.device
        if isinstance(default_dev, tuple) or (hasattr(default_dev, '__len__') and len(default_dev) == 2):
            default_input = default_dev[0]
        elif isinstance(default_dev, int):
            default_input = default_dev
    except:
        pass

    for idx, name, channels, api in devices:
        default_marker = " (Current)" if idx == default_input else ""
        # Truncate long names
        display_name = (name[:37] + '...') if len(name) > 37 else name
        print(f"[{idx:<2}] {display_name:<40} {api:<10}{default_marker}")
    print("-" * 60)
    return devices


def set_input_device(device_index):
    """Set the default input device."""
    # Get current output device to preserve it
    current = sd.default.device
    output_dev = None

    try:
        # Try to extract output device index (second element)
        output_dev = current[1]
    except (TypeError, IndexError):
        # If not indexable or too short, assume scalar (same for both)
        output_dev = current
    
    # Ensure output_dev is strictly an integer (device index)
    if not isinstance(output_dev, int):
        # Try to query the default output device as fallback
        try:
            output_dev = sd.query_devices(kind='output')['index']
        except:
            output_dev = None

    # Set input device while preserving output (must be int data)
    sd.default.device = (device_index, output_dev)
    
    try:
        # Try to open the stream briefly to test validity
        # This prevents setting a bad default that crashes later
        try:
            with sd.InputStream(device=device_index, samplerate=config.SAMPLE_RATE, channels=1):
                pass
            print(f"Input device set to: {sd.query_devices(device_index)['name']}")
        except Exception as e:
            print(f"WARNING: Device {device_index} selected, but failed connection test: {e}")
            print("It might need a different sample rate or is in use.")
    except:
        print(f"Input device set to index: {device_index}")


class AudioRecorder:
    """Records audio from microphone in a continuous stream."""

    def __init__(self, sample_rate=config.SAMPLE_RATE, channels=config.CHANNELS, device=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device if device is not None else config.MIC_DEVICE_INDEX
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self._stream = None

    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio stream."""
        if status:
            print(f"Audio status: {status}")

        # Convert to mono if stereo
        audio_data = indata.copy()
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            # Average channels to convert stereo to mono
            audio_data = np.mean(audio_data, axis=1, keepdims=False)

        # Ensure 1D array
        audio_data = audio_data.flatten()

        self.audio_queue.put(audio_data)

    def start(self):
        """Start recording audio."""
        self.is_recording = True
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=config.CHUNK_SIZE,
                device=self.device,
                callback=self._audio_callback
            )
            self._stream.start()
        except Exception as e:
            print(f"Error starting audio stream on device {self.device}: {e}")
            print("Attempting fallback to system default input device...")
            try:
                # Find the true system default input (usually MME 0 or similar) 
                # instead of relying on 'None' which might be set to the bad device
                default_input = sd.query_devices(kind='input')
                fallback_device = default_input['index']
                
                print(f"Trying fallback device: [{fallback_device}] {default_input['name']}")
                
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=np.float32,
                    blocksize=config.CHUNK_SIZE,
                    device=fallback_device, # Explicitly use the default index found
                    callback=self._audio_callback
                )
                self._stream.start()
                print("Fallback successful using system default device.")
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                self.is_recording = False
                raise e2

    def stop(self):
        """Stop recording audio."""
        self.is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_audio_chunk(self, timeout=1.0):
        """Get the next audio chunk from the queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_queue(self):
        """Clear any buffered audio."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break


def record_for_duration(duration_seconds, sample_rate=config.SAMPLE_RATE, device=None):
    """Record audio for a specific duration and return as numpy array."""
    frames = int(duration_seconds * sample_rate)
    dev = device if device is not None else config.MIC_DEVICE_INDEX
    recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype=np.float32, device=dev)
    sd.wait()
    return recording.flatten()


def record_until_silence(max_duration=config.LISTEN_TIMEOUT,
                         silence_threshold=0.01,
                         silence_duration=config.SILENCE_THRESHOLD,
                         sample_rate=config.SAMPLE_RATE,
                         device=None):
    """Record audio until silence is detected or max duration is reached."""
    chunks = []
    silence_chunks = 0
    chunks_for_silence = int(silence_duration * sample_rate / config.CHUNK_SIZE)
    max_chunks = int(max_duration * sample_rate / config.CHUNK_SIZE)

    recorder = AudioRecorder(sample_rate, device=device)
    recorder.start()

    try:
        chunk_count = 0
        while chunk_count < max_chunks:
            chunk = recorder.get_audio_chunk(timeout=1.0)
            if chunk is None:
                continue

            chunks.append(chunk)
            chunk_count += 1

            # Check for silence
            rms = np.sqrt(np.mean(chunk**2))
            if rms < silence_threshold:
                silence_chunks += 1
                min_chunks_before_silence = int(2 * sample_rate / config.CHUNK_SIZE)
                if silence_chunks >= chunks_for_silence and chunk_count > min_chunks_before_silence:
                    break
            else:
                silence_chunks = 0
    finally:
        recorder.stop()

    if chunks:
        return np.concatenate(chunks).flatten()
    return np.array([], dtype=np.float32)


def play_audio(audio_data, sample_rate=config.SAMPLE_RATE):
    """Play audio data through the speakers."""
    # Query the default output device directly to avoid _InputOutputPair issues
    try:
        default_output = sd.query_devices(kind='output')
        output_device = default_output['index'] if default_output else None
    except Exception:
        output_device = None

    sd.play(audio_data, samplerate=sample_rate, device=output_device)
    sd.wait()


def play_audio_file(file_path):
    """Play an audio file through the speakers."""
    data, sample_rate = sf.read(file_path)
    play_audio(data, sample_rate)


def get_audio_devices():
    """List available audio devices."""
    return sd.query_devices()
