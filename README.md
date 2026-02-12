<p align="center">
  <img src="Jarvis-banner.png" alt="Jarvis AI Assistant Banner" />
</p>

This is a Jarvis AI assistant that can be used to answer questions and perform tasks. It can do the following:

- Listen for the wake word 'Hey Jarvis' using openWakeWord (dedicated wake word detection)
- Transcribe speech to text using Distil-Whisper
- Generate responses using your choice of LLM. Qwen3-8B is recommended.
- Speak responses using Piper TTS with custom Jarvis voice

## System Requirements

- **Disk Space**: ~18 GB for models
- **RAM**: 8 GB minimum (16 GB recommended)
- **GPU**: Optional (CUDA-compatible GPU will improve performance)
- **OS**: Windows, Linux, or macOS

## Installation

### 1. Install Git LFS (for voice models)

```bash
git lfs install
```

### 2. Clone the repository

```bash
git clone https://github.com/Spakeyvr/Jarvis-AI-Assistant.git
cd Jarvis-AI-Assistant
```
(You can also download the .zip file and extract it in case you aren't experienced with Git)

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Qwen3-8B model
#### This step uses the recommended LLM, you can choose your own
The language model is not included in this repository due to its size (~16 GB). Download it from Hugging Face:

```bash
# Download the model (huggingface-hub is already in requirements.txt)
huggingface-cli download Qwen/Qwen2.5-8B-Instruct --local-dir Qwen3-8B
```

Alternatively, you can download it manually from [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-8B-Instruct) and place it in a folder named `Qwen3-8B` in the project root.

### 4.5. Verify your installation (optional but recommended)

```bash
python verify_setup.py
```

This will check that all dependencies are installed and models are in place.

### 5. Run the assistant

```bash
# Windows
run.bat

# Linux/Mac
python jarvis/main.py
```

**On first run**, the following models will be automatically downloaded:
- Distil-Whisper (~750 MB) - for speech recognition
- openWakeWord models (~15 MB) - for wake word detection
- Piper voice models (if not already present)

This may take a few minutes depending on your internet connection.

## Features & Limitations

**What it can do:**
- Wake word detection with "Hey Jarvis"
- Voice-activated question answering
- Follow-up conversation mode (2-second window after responses)
- Completely offline operation after initial setup

**Important limitations:**
- No real-time data access
- No internet access (offline only)
- No screen/camera access
- No file system access
- No data sent to external servers
- Internet only used for initial model downloads

## Configuration

You can customize Jarvis by editing `jarvis/config.py`:

### Wake Word Detection
```python
# Adjust wake word sensitivity (0.0-1.0)
OPENWAKEWORD_THRESHOLD = 0.5  # Lower = more sensitive, higher = stricter

# Enable debug mode to see detection scores
WAKE_WORD_DEBUG = True
```

### Follow-up Conversation
```python
# Enable/disable follow-up questions without saying "Hey Jarvis"
CONTINUE_CONVERSATION_ENABLED = True

# Time window for follow-up questions (seconds)
CONTINUE_CONVERSATION_TIMEOUT = 2
```

### Microphone Selection
On first run, you'll be prompted to select your microphone. You can also set it permanently:
```python
MIC_DEVICE_INDEX = 22  # Set to your microphone's device index
```

## Troubleshooting

**Wake word not detected consistently:**
- Lower the detection threshold in `config.py`: `OPENWAKEWORD_THRESHOLD = 0.3`
- Enable debug mode to see detection scores: `WAKE_WORD_DEBUG = True`
- Ensure you're speaking clearly and saying "Hey Jarvis" distinctly

**Too many false wake word detections:**
- Increase the detection threshold: `OPENWAKEWORD_THRESHOLD = 0.6`
- Reduce background noise

**Questions not being captured:**
- Speak immediately after the acknowledgment sound
- Check microphone levels in your system settings
- Adjust `LISTEN_TIMEOUT` and `SILENCE_THRESHOLD` in `config.py`
