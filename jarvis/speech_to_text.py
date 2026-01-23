"""Speech-to-text using Distil-Whisper for transcription."""

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import config
from audio_utils import record_until_silence


class SpeechToText:
    """Transcribes audio to text using Distil-Whisper."""

    def __init__(self, model_id=config.WHISPER_MODEL):
        """Initialize the Distil-Whisper model."""
        print(f"Loading Distil-Whisper model ({model_id})...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        self.sample_rate = config.SAMPLE_RATE
        print("Distil-Whisper model loaded.")

    def transcribe_audio(self, audio_data, sample_rate=config.SAMPLE_RATE):
        """
        Transcribe audio data to text.

        Args:
            audio_data: Numpy array of audio samples (float32)
            sample_rate: Sample rate of the audio

        Returns:
            Transcribed text string
        """
        if len(audio_data) == 0:
            return ""

        # Ensure audio is float32
        audio_data = audio_data.astype(np.float32)

        # Transcribe using pipeline
        result = self.pipe(
            {"raw": audio_data, "sampling_rate": sample_rate},
            generate_kwargs={"language": "english"}
        )

        return result["text"].strip()

    def listen_and_transcribe(self, max_duration=config.LISTEN_TIMEOUT):
        """
        Listen to microphone and transcribe the speech.

        Args:
            max_duration: Maximum seconds to listen

        Returns:
            Transcribed text string
        """
        print("Listening for your question...")
        audio_data = record_until_silence(max_duration=max_duration)

        if len(audio_data) == 0:
            print("No audio captured.")
            return ""

        print("Transcribing...")
        text = self.transcribe_audio(audio_data)
        print(f"Transcribed: '{text}'")
        return text


if __name__ == "__main__":
    # Test speech-to-text
    stt = SpeechToText()
    print("Speak something to test transcription...")
    text = stt.listen_and_transcribe()
    print(f"You said: {text}")
