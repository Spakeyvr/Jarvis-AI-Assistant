"""LLM handler with configurable quantization and optional multimodal support."""

from datetime import datetime
import re
import threading
import torch

# Enable TF32 for RTX 4070 Ti - improves performance with minimal precision loss
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from transformers import BitsAndBytesConfig, TextIteratorStreamer
import config

if config.MULTIMODAL:
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from PIL import Image
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        """Load the LLM model with configured quantization."""
        self.multimodal = config.MULTIMODAL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        quant_mode = config.LLM_QUANTIZATION.lower()
        requested_quant_mode = quant_mode
        print(f"Loading LLM model with {quant_mode} quantization on {self.device}...")

        if self.device != "cuda" and quant_mode in {"4bit", "8bit"}:
            print(
                f"Warning: {quant_mode} quantization requires CUDA. "
                "Falling back to CPU full precision."
            )
            quant_mode = "none"

        # Configure quantization when CUDA is available.
        quantization_config = None
        if quant_mode == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif quant_mode == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        load_kwargs = {
            "trust_remote_code": True
        }
        if self.device == "cuda":
            load_kwargs["device_map"] = {"": "cuda"}
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32

        if self.multimodal:
            # Multimodal: use AutoProcessor + Qwen3_5ForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(
                str(config.LLM_MODEL_PATH),
                trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer

            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                str(config.LLM_MODEL_PATH),
                **load_kwargs
            )
        else:
            # Text-only: use AutoTokenizer + AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(config.LLM_MODEL_PATH),
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                str(config.LLM_MODEL_PATH),
                **load_kwargs
            )

        # Clear sampling params from generation config since we use do_sample=False
        # This prevents warnings about unused temperature/top_p/top_k
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

        # Initialize conversation history
        self.conversation_history = []
        self.screenshot_requested = False

        mode_str = "multimodal" if self.multimodal else "text-only"
        effective_quant_mode = requested_quant_mode if quantization_config else quant_mode
        print(
            f"LLM model loaded successfully ({effective_quant_mode}, {mode_str}, device={self.device})."
        )

    def _count_tokens(self, messages):
        """Count tokens in a list of messages."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        return len(self.tokenizer.encode(text))

    def _trim_history(self):
        """Trim conversation history aggressively for voice assistant use case.

        For VRAM stability, keep only the most recent turn (1 user + 1 assistant message).
        This prevents KV-cache accumulation over long sessions.
        """
        # For voice assistant: keep only current turn to prevent KV-cache blowup
        # Only keep the last user message (current question) - previous turns are discarded
        if len(self.conversation_history) > 1:
            # Keep only the last message (current user question)
            self.conversation_history = self.conversation_history[-1:]

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")

    def should_capture_screenshot(self, question):
        """Fast-path obvious visual requests to avoid an extra model round trip."""
        if not self.multimodal or not question:
            return False

        normalized = question.strip().lower()
        return any(re.search(pattern, normalized) for pattern in SCREENSHOT_INTENT_PATTERNS)

    def generate_response(self, question, image=None):
        """
        Generate a response to the user's question.

        Args:
            question: The user's question string
            image: Optional PIL Image for multimodal input

        Returns:
            Generated response string
        """
        self.screenshot_requested = False

        # Build user message content
        if self.multimodal and image:
            user_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": question + " /no_think"}
            ]
        else:
            user_content = question + " /no_think"

        # Add user message to history (text-only for history to save memory)
        self.conversation_history.append({
            "role": "user",
            "content": question + " /no_think"
        })

        # Trim history if needed to fit context window
        self._trim_history()

        # Build system prompt with current date/time
        now = datetime.now()
        datetime_info = now.strftime("Current date: %A, %B %d, %Y. Current time: %I:%M %p.")
        system_content = f"{config.SYSTEM_PROMPT}\n\n{datetime_info}"

        # Append screenshot instruction when multimodal is enabled but no image yet
        if self.multimodal and not image:
            system_content += SCREENSHOT_INSTRUCTION

        # Build the prompt with system message + conversation history
        # Use the image-bearing content for the current (last) user message
        messages = [
            {"role": "system", "content": system_content}
        ] + self.conversation_history[:-1]  # all history except current

        # Add current user message with potential image content
        messages.append({
            "role": "user",
            "content": user_content
        })

        if self.multimodal and image:
            # Multimodal path: use process_vision_info + processor
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                ),
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt"
            ).to(self.model.device)
        else:
            # Text-only path
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate response - deterministic decoding for speed and VRAM stability
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )

        # Decode response (only the new tokens)
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Strip any thinking tags that might still appear
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = response.strip()

        # Check if model requested a screenshot
        if SCREENSHOT_TAG in response:
            self.screenshot_requested = True
            # Remove the user message from history so it can be re-added on re-query
            self.conversation_history.pop()
            return ""

        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def generate_response_streaming(self, question, image=None):
        """
        Generate a response to the user's question with streaming output.

        Args:
            question: The user's question string
            image: Optional PIL Image for multimodal input

        Yields:
            Text chunks as they are generated
        """
        self.screenshot_requested = False

        # Build user message content
        if self.multimodal and image:
            user_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": question + " /no_think"}
            ]
        else:
            user_content = question + " /no_think"

        # Add user message to history (text-only for history to save memory)
        self.conversation_history.append({
            "role": "user",
            "content": question + " /no_think"
        })

        # Trim history if needed to fit context window
        self._trim_history()

        # Build system prompt with current date/time
        now = datetime.now()
        datetime_info = now.strftime("Current date: %A, %B %d, %Y. Current time: %I:%M %p.")
        system_content = f"{config.SYSTEM_PROMPT}\n\n{datetime_info}"

        # Append screenshot instruction when multimodal is enabled but no image yet
        if self.multimodal and not image:
            system_content += SCREENSHOT_INSTRUCTION

        # Build the prompt with system message + conversation history
        messages = [
            {"role": "system", "content": system_content}
        ] + self.conversation_history[:-1]

        # Add current user message with potential image content
        messages.append({
            "role": "user",
            "content": user_content
        })

        if self.multimodal and image:
            # Multimodal path: use process_vision_info + processor
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                ),
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt"
            ).to(self.model.device)

            # Streamer uses the processor's tokenizer for multimodal
            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_special_tokens=True,
                skip_prompt=True
            )
        else:
            # Text-only path
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_special_tokens=True,
                skip_prompt=True
            )

        # Generation kwargs - deterministic decoding for speed and VRAM stability
        generation_kwargs = {
            **inputs,
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "streamer": streamer
        }

        # Run generation in background thread
        thread = threading.Thread(target=self._generate_with_kwargs, args=(generation_kwargs,))
        thread.start()

        # Collect full response for history while yielding chunks
        full_response = ""

        # Buffer initial tokens to detect screenshot tag before yielding
        buffer = ""
        buffer_limit = len(SCREENSHOT_TAG) + 5
        buffering = self.multimodal and not image

        # Yield tokens as they arrive
        for text in streamer:
            # Filter out any thinking tags
            if "<think>" not in text and "</think>" not in text:
                if buffering:
                    buffer += text
                    if len(buffer) >= buffer_limit:
                        # Buffer full, check for screenshot tag
                        if SCREENSHOT_TAG in buffer:
                            self.screenshot_requested = True
                            thread.join()
                            # Remove user message so it can be re-added on re-query
                            self.conversation_history.pop()
                            return
                        # No tag found, flush buffer and stop buffering
                        buffering = False
                        full_response += buffer
                        yield buffer
                else:
                    full_response += text
                    yield text

        thread.join()

        # Check buffer remainder (response was shorter than buffer limit)
        if buffering and buffer:
            if SCREENSHOT_TAG in buffer:
                self.screenshot_requested = True
                self.conversation_history.pop()
                return
            # No tag, flush whatever we have
            full_response += buffer
            yield buffer

        # Clean up the full response and add to history
        full_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL)
        full_response = full_response.strip()

        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": full_response
        })

    def _generate_with_kwargs(self, kwargs):
        """Helper to run generation with kwargs in a thread."""
        with torch.no_grad():
            self.model.generate(**kwargs)


if __name__ == "__main__":
    # Test LLM handler
    llm = LLMHandler()

    test_questions = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "Who wrote Romeo and Juliet?"
    ]

    for question in test_questions:
        print(f"\nQ: {question}")
        response = llm.generate_response(question)
        print(f"A: {response}")
