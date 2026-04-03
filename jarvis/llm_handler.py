"""LLM handler supporting HuggingFace and GGUF backends."""

from datetime import datetime
import re
import threading
import config

SCREENSHOT_TAG = "[SCREENSHOT]"
SCREENSHOT_INSTRUCTION = (
    "\n\nYou have the ability to see the user's screen. "
    "If the user asks about what's on their screen, asks you to look at or read something, "
    "or any request that requires visual information, respond with exactly [SCREENSHOT] "
    "and nothing else. You will then be shown a screenshot to answer their question."
)
SCREENSHOT_INTENT_PATTERNS = (
    r"\bon my screen\b",
    r"\bmy screen\b",
    r"\bon screen\b",
    r"\blook at (?:my )?screen\b",
    r"\bcheck (?:my )?screen\b",
    r"\bsee (?:my )?screen\b",
    r"\bwhat(?:'s| is) on (?:my )?screen\b",
    r"\bwhat do you see\b",
    r"\bwhat am i looking at\b",
    r"\bread this\b",
    r"\bcan you read this\b",
    r"\bwhat does this say\b",
    r"\bdescribe this\b",
    r"\bsummarize this\b",
)


class LLMHandler:
    """Handles inference with the local LLM model."""

    def __init__(self):
        self.backend = config.LLM_BACKEND.lower()

        # Multimodal is only supported with the HuggingFace backend
        if config.MULTIMODAL and self.backend == "gguf":
            print("Warning: Multimodal is not supported with the GGUF backend. Disabling.")
        self.multimodal = config.MULTIMODAL and self.backend == "huggingface"

        if self.backend == "gguf":
            self._load_gguf()
        else:
            self._load_huggingface()

        self.conversation_history = []
        self.screenshot_requested = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_gguf(self):
        """Load a GGUF model via llama-cpp-python."""
        from llama_cpp import Llama
        print(f"Loading GGUF model from {config.GGUF_MODEL_PATH}...")
        self.model = Llama(
            model_path=str(config.GGUF_MODEL_PATH),
            n_ctx=config.CONTEXT_WINDOW_SIZE,
            n_gpu_layers=config.GGUF_N_GPU_LAYERS,
            verbose=False,
        )
        print(f"GGUF model loaded (n_gpu_layers={config.GGUF_N_GPU_LAYERS}).")

    def _load_huggingface(self):
        """Load a HuggingFace model with optional BitsAndBytes quantization."""
        import torch
        from transformers import BitsAndBytesConfig, TextIteratorStreamer

        # Enable TF32 for better performance on Ampere+ GPUs
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Store references needed later by generation methods
        self._torch = torch
        self._TextIteratorStreamer = TextIteratorStreamer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        quant_mode = config.LLM_QUANTIZATION.lower()
        requested_quant_mode = quant_mode
        print(f"Loading HuggingFace model with {quant_mode} quantization on {self.device}...")

        if self.device != "cuda" and quant_mode in {"4bit", "8bit"}:
            print(
                f"Warning: {quant_mode} quantization requires CUDA. "
                "Falling back to CPU full precision."
            )
            quant_mode = "none"

        quantization_config = None
        if quant_mode == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quant_mode == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        load_kwargs = {"trust_remote_code": True}
        if self.device == "cuda":
            load_kwargs["device_map"] = {"": "cuda"}
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32

        if self.multimodal:
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(
                str(config.LLM_MODEL_PATH), trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                str(config.LLM_MODEL_PATH), **load_kwargs
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(config.LLM_MODEL_PATH), trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(config.LLM_MODEL_PATH), **load_kwargs
            )

        # Clear unused sampling params to suppress warnings
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

        effective_quant = requested_quant_mode if quantization_config else quant_mode
        mode_str = "multimodal" if self.multimodal else "text-only"
        print(f"HuggingFace model loaded ({effective_quant}, {mode_str}, device={self.device}).")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_messages(self, question, image=None):
        """Append the user turn to history, trim, then return the full messages list."""
        # /no_think disables chain-of-thought on Qwen3 models (HuggingFace only)
        history_content = question + (" /no_think" if self.backend == "huggingface" else "")
        self.conversation_history.append({"role": "user", "content": history_content})
        self._trim_history()

        now = datetime.now()
        datetime_info = now.strftime("Current date: %A, %B %d, %Y. Current time: %I:%M %p.")
        system_content = f"{config.SYSTEM_PROMPT}\n\n{datetime_info}"
        if self.multimodal and not image:
            system_content += SCREENSHOT_INSTRUCTION

        # Build user content for the current turn (may include image)
        if self.multimodal and image:
            current_user_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": question + " /no_think"},
            ]
        else:
            current_user_content = history_content

        messages = [{"role": "system", "content": system_content}]
        messages += self.conversation_history[:-1]  # prior turns
        messages.append({"role": "user", "content": current_user_content})
        return messages

    def _trim_history(self):
        """Keep only the most recent turn to prevent KV-cache / context blowup."""
        if len(self.conversation_history) > 1:
            self.conversation_history = self.conversation_history[-1:]

    def clear_history(self):
        self.conversation_history = []
        print("Conversation history cleared.")

    def should_capture_screenshot(self, question):
        """Fast-path obvious visual requests to avoid an extra model round trip."""
        if not self.multimodal or not question:
            return False
        normalized = question.strip().lower()
        return any(re.search(p, normalized) for p in SCREENSHOT_INTENT_PATTERNS)

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate_response(self, question, image=None):
        self.screenshot_requested = False
        messages = self._build_messages(question, image)
        if self.backend == "gguf":
            return self._generate_gguf(messages)
        return self._generate_huggingface(messages, image)

    def generate_response_streaming(self, question, image=None):
        self.screenshot_requested = False
        messages = self._build_messages(question, image)
        if self.backend == "gguf":
            yield from self._generate_gguf_streaming(messages)
        else:
            yield from self._generate_huggingface_streaming(messages, image)

    # ------------------------------------------------------------------
    # GGUF backend
    # ------------------------------------------------------------------

    def _generate_gguf(self, messages):
        result = self.model.create_chat_completion(
            messages=messages,
            max_tokens=config.MAX_NEW_TOKENS,
            temperature=0,
        )
        response = result["choices"][0]["message"]["content"].strip()
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def _generate_gguf_streaming(self, messages):
        stream = self.model.create_chat_completion(
            messages=messages,
            max_tokens=config.MAX_NEW_TOKENS,
            temperature=0,
            stream=True,
        )
        full_response = ""
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            content = delta.get("content")
            if content:
                full_response += content
                yield content

        self.conversation_history.append({"role": "assistant", "content": full_response.strip()})

    # ------------------------------------------------------------------
    # HuggingFace backend
    # ------------------------------------------------------------------

    def _generate_huggingface(self, messages, image=None):
        torch = self._torch

        if self.multimodal and image:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                ),
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            ).to(self.model.device)
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        if SCREENSHOT_TAG in response:
            self.screenshot_requested = True
            self.conversation_history.pop()
            return ""

        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def _generate_huggingface_streaming(self, messages, image=None):
        torch = self._torch
        TextIteratorStreamer = self._TextIteratorStreamer

        if self.multimodal and image:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                ),
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            ).to(self.model.device)
            streamer = TextIteratorStreamer(
                self.processor.tokenizer, skip_special_tokens=True, skip_prompt=True
            )
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_special_tokens=True, skip_prompt=True
            )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "streamer": streamer,
        }

        thread = threading.Thread(target=self._run_generation, args=(generation_kwargs,))
        thread.start()

        full_response = ""
        buffer = ""
        buffer_limit = len(SCREENSHOT_TAG) + 5
        buffering = self.multimodal and not image

        for text in streamer:
            if "<think>" in text or "</think>" in text:
                continue
            if buffering:
                buffer += text
                if len(buffer) >= buffer_limit:
                    if SCREENSHOT_TAG in buffer:
                        self.screenshot_requested = True
                        thread.join()
                        self.conversation_history.pop()
                        return
                    buffering = False
                    full_response += buffer
                    yield buffer
            else:
                full_response += text
                yield text

        thread.join()

        if buffering and buffer:
            if SCREENSHOT_TAG in buffer:
                self.screenshot_requested = True
                self.conversation_history.pop()
                return
            full_response += buffer
            yield buffer

        full_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
        self.conversation_history.append({"role": "assistant", "content": full_response})

    def _run_generation(self, kwargs):
        with self._torch.no_grad():
            self.model.generate(**kwargs)


if __name__ == "__main__":
    llm = LLMHandler()
    for question in ["What is the capital of France?", "What is 2 + 2?", "Who wrote Romeo and Juliet?"]:
        print(f"\nQ: {question}")
        print(f"A: {llm.generate_response(question)}")
