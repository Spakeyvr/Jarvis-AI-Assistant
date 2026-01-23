"""Configuration settings for Jarvis AI."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = BASE_DIR / "models"

# Qwen3-8B model path
LLM_MODEL_PATH = PROJECT_ROOT / "Qwen3-8B"

# Audio settings
SAMPLE_RATE = 16000  # Required for Vosk and Whisper
CHANNELS = 1
CHUNK_SIZE = 8000  # Audio chunk size for processing (0.5 seconds at 16kHz)
MIC_DEVICE_INDEX = None  # None = use system default, or set to device index number

# Wake word settings
WAKE_PHRASE = "hey jarvis"
WAKE_WORD_ENGINE = "openwakeword"  # Options: "openwakeword" (recommended), "whisper" (legacy)

# openWakeWord settings (recommended - dedicated wake word engine)
OPENWAKEWORD_THRESHOLD = 0.5  # Detection threshold (0.0-1.0). Lower = more sensitive, higher = more strict
WAKE_WORD_DEBUG = False  # set to True to see wake word scores in real-time

# Legacy Whisper wake word settings (only used if WAKE_WORD_ENGINE = "whisper")
WAKE_WORD_ENERGY_THRESHOLD = 0.005  # Minimum audio energy to process for wake word (lowered for better sensitivity)
WAKE_WORD_SENSITIVITY = "medium"  # Options: "low", "medium", "high" - controls detection strictness

# Conversation continuation settings
# When enabled, you can continue talking without saying "Hey Jarvis" for a few seconds after a response
CONTINUE_CONVERSATION_ENABLED = True
CONTINUE_CONVERSATION_TIMEOUT = 2  # Seconds to wait for follow-up before returning to wake word mode

# Speech-to-text settings (Distil-Whisper)
WHISPER_MODEL = "distil-whisper/distil-large-v3"  # ~750M params, fast & accurate
LISTEN_TIMEOUT = 5  # Seconds to listen for question after wake word
SILENCE_THRESHOLD = 0.8  # Seconds of silence to end recording

# LLM settings
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.6
CONTEXT_WINDOW_SIZE = 4096  # Max tokens to keep in conversation history
SYSTEM_PROMPT = """You are Jarvis, a helpful AI assistant. Provide brief, concise answers.
Keep responses to 1-2 sentences maximum. Be direct and informative."""

# Text-to-speech settings
PIPER_VOICE = "jarvis-high"  # Custom Jarvis voice (options: jarvis-high, jarvis-medium)
PIPER_MODEL_DIR = MODELS_DIR / "piper"

# Custom Jarvis voice paths
JARVIS_VOICE_DIR = PROJECT_ROOT / "jarvis-voice" / "en" / "en_GB" / "jarvis"
JARVIS_VOICES = {
    "jarvis-high": JARVIS_VOICE_DIR / "high",
    "jarvis-medium": JARVIS_VOICE_DIR / "medium", # This Jarvis voice will be replaced by a new & better voice in the future
}

# Create directories if they don't exist
os.makedirs(PIPER_MODEL_DIR, exist_ok=True)
