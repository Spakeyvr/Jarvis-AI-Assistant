"""LLM handler with configurable quantization."""

from datetime import datetime
import threading
import torch

# Enable TF32 for RTX 4070 Ti - improves performance with minimal precision loss
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
import config


class LLMHandler:
    """Handles inference with the local LLM model."""

    def __init__(self):
        """Load the LLM model with configured quantization."""
        quant_mode = config.LLM_QUANTIZATION.lower()
        print(f"Loading LLM model with {quant_mode} quantization...")

        # Configure quantization based on config - GPU-only, no CPU offload
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

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(config.LLM_MODEL_PATH),
            trust_remote_code=True
        )

        # Load model - force GPU-only execution (no CPU offload)
        load_kwargs = {
            "device_map": {"": "cuda"},
            "trust_remote_code": True
        }
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
        else:
            # Full precision
            load_kwargs["torch_dtype"] = torch.float16

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

        print(f"LLM model loaded successfully ({quant_mode}).")

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

    def generate_response(self, question):
        """
        Generate a response to the user's question.

        Args:
            question: The user's question string

        Returns:
            Generated response string
        """
        import re

        # Add user message to history (with /no_think to disable thinking mode)
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

        # Build the prompt with system message + conversation history
        messages = [
            {"role": "system", "content": system_content}
        ] + self.conversation_history

        # Apply chat template with thinking disabled
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Disable thinking mode
        )

        # Tokenize
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

        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def generate_response_streaming(self, question):
        """
        Generate a response to the user's question with streaming output.

        Args:
            question: The user's question string

        Yields:
            Text chunks as they are generated
        """
        import re

        # Add user message to history (with /no_think to disable thinking mode)
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

        # Build the prompt with system message + conversation history
        messages = [
            {"role": "system", "content": system_content}
        ] + self.conversation_history

        # Apply chat template with thinking disabled
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Setup streamer for real-time token output
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

        # Yield tokens as they arrive
        for text in streamer:
            # Filter out any thinking tags
            if "<think>" not in text and "</think>" not in text:
                full_response += text
                yield text

        thread.join()

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
