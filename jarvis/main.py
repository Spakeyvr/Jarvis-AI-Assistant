"""
Jarvis AI - Voice-activated AI Assistant

Main entry point that runs the Jarvis assistant in the background,
listening for 'Hey Jarvis' wake word and responding to questions.
"""

import sys
import signal
import time


class Jarvis:
    """Main Jarvis AI assistant class."""

    def __init__(self):
        """Initialize all Jarvis components."""
        print("=" * 50)
        print("Initializing Jarvis AI...")
        print("=" * 50)

        # Import and initialize components
        print("\n[1/4] Loading wake word detector...")
        from wake_word import WakeWordDetector
        self.wake_word = WakeWordDetector()

        print("\n[2/4] Loading speech-to-text (Distil-Whisper)...")
        from speech_to_text import SpeechToText
        self.stt = SpeechToText()

        print("\n[3/4] Loading LLM (Qwen3-8B)...")
        from llm_handler import LLMHandler
        self.llm = LLMHandler()

        print("\n[4/4] Loading text-to-speech (Piper)...")
        from text_to_speech import TextToSpeech
        self.tts = TextToSpeech()

        self.running = False 
        print("\n" + "=" * 50)
        print("Jarvis AI initialized successfully!")
        print("=" * 50)

    def process_question(self, question):
        """
        Process a question and generate a spoken response.

        Args:
            question: The transcribed question from the user
        """
        if not question or len(question.strip()) < 2:
            self.tts.speak("I didn't catch that. Could you please repeat?")
            return

        print(f"\nProcessing question: '{question}'")

        # Generate response using LLM
        response = self.llm.generate_response(question)

        # Speak the response
        self.tts.speak(response)

    def run(self):
        """Main loop - listen for wake word and process questions."""
        import config

        self.running = True

        # Greeting
        self.tts.speak("Jarvis online. How may I assist you?")

        print("\n" + "-" * 50)
        print("Jarvis is now listening...")
        print("Say 'Hey Jarvis' followed by your question.")
        if config.CONTINUE_CONVERSATION_ENABLED:
            print(f"(Follow-up mode: {config.CONTINUE_CONVERSATION_TIMEOUT}s window after each response)")
        print("Press Ctrl+C to exit.")
        print("-" * 50 + "\n")

        self.wake_word.start_listening()

        try:
            while self.running:
                # Check for wake word
                if self.wake_word.check_for_wake_word():
                    # Wake word detected - play acknowledgment
                    print("\nWake word detected!")

                    # Stop wake word listening temporarily
                    self.wake_word.stop_listening()

                    # Listen for the question
                    question = self.stt.listen_and_transcribe()

                    # Process and respond
                    self.process_question(question)

                    # Check if conversation continuation is enabled
                    if config.CONTINUE_CONVERSATION_ENABLED:
                        # Allow follow-up questions without wake word
                        self._handle_follow_up_conversation()

                    # Resume wake word listening
                    print("\nListening for 'Hey Jarvis'...")
                    self.wake_word.start_listening()

        except KeyboardInterrupt:
            print("\n\nShutting down Jarvis...")
        finally:
            self.shutdown()

    def _handle_follow_up_conversation(self):
        """Handle follow-up questions without requiring wake word."""
        import config

        while self.running:
            print(f"\n(Listening for follow-up... {config.CONTINUE_CONVERSATION_TIMEOUT}s window)")

            # Listen for follow-up with timeout
            question = self.stt.listen_and_transcribe(
                max_duration=config.CONTINUE_CONVERSATION_TIMEOUT
            )

            # If no speech detected or empty, exit follow-up mode
            if not question or len(question.strip()) < 2:
                print("No follow-up detected, returning to wake word mode.")
                break

            # Process the follow-up question
            self.process_question(question)

    def shutdown(self):
        """Clean shutdown of Jarvis."""
        self.running = False
        self.wake_word.stop_listening()
        self.tts.speak("Goodbye.")
        print("Jarvis has been shut down.")


def select_microphone():
    """Let user select a microphone if needed."""
    from audio_utils import print_input_devices, set_input_device
    import config

    devices = print_input_devices()

    if config.MIC_DEVICE_INDEX is not None:
        print(f"\nUsing configured mic: [{config.MIC_DEVICE_INDEX}]")
        return

    print("\nPress Enter to use default mic, or enter device number: ", end="")
    try:
        choice = input().strip()
        if choice:
            device_idx = int(choice)
            # Verify it's a valid input device
            valid_ids = [d[0] for d in devices]
            if device_idx in valid_ids:
                set_input_device(device_idx)
                config.MIC_DEVICE_INDEX = device_idx
            else:
                print(f"Invalid device {device_idx}, using default.")
    except ValueError:
        print("Using default microphone.")
    except EOFError:
        print("Using default microphone.")


def main():
    """Entry point for Jarvis AI."""
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nReceived shutdown signal...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Let user select microphone
    select_microphone()

    # Create and run Jarvis
    jarvis = Jarvis()
    jarvis.run()


if __name__ == "__main__":
    main()
