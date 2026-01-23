This is a Jarvis AI assistant that can be used to answer questions and perform tasks. It can do the following:

- Listen for the wake word 'Hey Jarvis'
- Transcribe speech to text using Distil-Whisper
- Generate responses using Qwen3-8B
- Speak responses using Piper

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

The language model is not included in this repository due to its size (~16 GB). Download it from Hugging Face:

```bash
# Install huggingface-cli if you don't have it
pip install huggingface-hub

# Download the model
huggingface-cli download Qwen/Qwen2.5-8B-Instruct --local-dir Qwen3-8B
```

Alternatively, you can download it manually from [Hugging Face](https://huggingface.co/Qwen/Qwen3-8B) and place it in a folder named `Qwen3-8B` in the project root.

### 5. Run the assistant

```bash
# Windows
run.bat

# Linux/Mac
python jarvis/main.py
```

Important:
- It does not have real-time data access.
- It is not able to access the internet.
- It is not able to see your screen.
- It is not able to access any files.
- It is not able to access your camera.
- There is no data being sent to any servers.
- The only times it will access the internet is when you first run it and it downloads the required models.
