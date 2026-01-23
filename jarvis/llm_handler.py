"""LLM handler for Qwen3-8B with 4-bit quantization."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import config


class LLMHandler:
    """Handles inference with the local Qwen3-8B model."""

    def __init__(self):
        """Load the Qwen3-8B model with 4-bit quantization."""
        print("Loading Qwen3-8B model with 4-bit quantization...")

        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(config.LLM_MODEL_PATH),
            trust_remote_code=True
        )

        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            str(config.LLM_MODEL_PATH),
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Initialize conversation history
        self.conversation_history = []

        print("Qwen3-8B model loaded successfully.")

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
        """Trim conversation history to fit within context window."""
        # Always keep system prompt separate - it's added fresh each time
        # We only track user/assistant message pairs in history

        while self.conversation_history:
            # Build test messages with system prompt + history
            test_messages = [
                {"role": "system", "content": config.SYSTEM_PROMPT}
            ] + self.conversation_history

            token_count = self._count_tokens(test_messages)

            if token_count <= config.CONTEXT_WINDOW_SIZE:
                break

            # Remove oldest user/assistant pair (first 2 messages)
            if len(self.conversation_history) >= 2:
                self.conversation_history = self.conversation_history[2:]
            else:
                self.conversation_history = []
                break

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

        # Build the prompt with system message + conversation history
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT}
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

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                do_sample=True,
                top_p=0.9,
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
