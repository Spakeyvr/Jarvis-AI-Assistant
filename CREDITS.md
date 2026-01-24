# Credits & Attribution

Jarvis is built using the following open-source AI models and libraries. A huge thank you to all the developers and researchers who made this project possible.

---

## AI Models

### Qwen3-8B
- **Purpose:** Large Language Model for conversation and reasoning
- **Creator:** Alibaba Cloud / Qwen Team
- **License:** Apache 2.0
- **Links:**
  - [Hugging Face](https://huggingface.co/Qwen/Qwen3-8B)
  - [GitHub](https://github.com/QwenLM/Qwen3)
  - [Paper](https://arxiv.org/abs/2505.09388)

### Distil-Whisper (distil-large-v3)
- **Purpose:** Speech-to-Text transcription
- **Creator:** Hugging Face (distilled from OpenAI Whisper)
- **License:** MIT
- **Links:**
  - [Hugging Face](https://huggingface.co/distil-whisper/distil-large-v3)
  - [GitHub](https://github.com/huggingface/distil-whisper)
  - [Paper](https://arxiv.org/abs/2311.00430)

### openWakeWord
- **Purpose:** Wake word detection ("Hey Jarvis")
- **Creator:** David Scripka
- **License:** Apache 2.0
- **Links:**
  - [GitHub](https://github.com/dscripka/openWakeWord)
  - [PyPI](https://pypi.org/project/openwakeword/)

### Piper TTS
- **Purpose:** Text-to-Speech synthesis
- **Creator:** Rhasspy / Michael Hansen
- **License:** MIT
- **Links:**
  - [GitHub](https://github.com/rhasspy/piper)
  - [Hugging Face Voices](https://huggingface.co/rhasspy/piper-voices)
  - [Website](https://rhasspy.github.io/piper-samples/)

---

## Core Libraries

### PyTorch
- **Purpose:** Deep learning framework
- **License:** BSD-3-Clause
- **Link:** [pytorch.org](https://pytorch.org/)

### Transformers
- **Purpose:** Model loading and inference
- **Creator:** Hugging Face
- **License:** Apache 2.0
- **Link:** [GitHub](https://github.com/huggingface/transformers)

### Accelerate
- **Purpose:** Hardware acceleration and optimization
- **Creator:** Hugging Face
- **License:** Apache 2.0
- **Link:** [GitHub](https://github.com/huggingface/accelerate)

### BitsAndBytes
- **Purpose:** 4-bit quantization for reduced memory usage
- **Creator:** Tim Dettmers
- **License:** MIT
- **Link:** [GitHub](https://github.com/TimDettmers/bitsandbytes)

---

## Audio Libraries

### SoundDevice
- **Purpose:** Audio recording and playback
- **License:** MIT
- **Link:** [GitHub](https://github.com/spatialaudio/python-sounddevice)

### SoundFile
- **Purpose:** Audio file reading/writing
- **License:** BSD-3-Clause
- **Link:** [GitHub](https://github.com/bastibe/python-soundfile)

---

## Other Dependencies

| Library | Purpose | License |
|---------|---------|---------|
| NumPy | Numerical computing | BSD-3-Clause |
| SciPy | Scientific computing | BSD-3-Clause |
| keyboard | Global hotkey support | MIT |
| huggingface-hub | Model downloading | Apache 2.0 |

---

## Acknowledgments

Special thanks to:
- The open-source AI community for making powerful models freely available
- Alibaba Cloud for the Qwen series of language models
- Hugging Face for their incredible ecosystem and Distil-Whisper
- The Rhasspy project for high-quality offline TTS
- All contributors to the libraries that make this project possible

---

*This project is made possible by the hard work of countless open-source developers and researchers. If you use Jarvis, consider starring or supporting the projects listed above.*
