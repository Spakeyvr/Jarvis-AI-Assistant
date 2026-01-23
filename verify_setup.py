#!/usr/bin/env python3
"""
Verify Jarvis AI setup and installation.
Run this script after installation to check if everything is configured correctly.
"""

import sys
import importlib.util
from pathlib import Path

def check_module(name, package=None):
    """Check if a Python module is installed."""
    pkg = package or name
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"❌ {pkg} is NOT installed")
        return False
    else:
        print(f"✅ {pkg} is installed")
        return True

def check_file(path, description):
    """Check if a file or directory exists."""
    if path.exists():
        print(f"✅ {description} found at: {path}")
        return True
    else:
        print(f"❌ {description} NOT found at: {path}")
        return False

def main():
    print("=" * 60)
    print("Jarvis AI Setup Verification")
    print("=" * 60)

    all_good = True

    # Check Python version
    print("\n[1/5] Checking Python version...")
    py_version = sys.version_info
    if py_version >= (3, 9):
        print(f"✅ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print(f"❌ Python {py_version.major}.{py_version.minor}.{py_version.micro} (3.9+ required)")
        all_good = False

    # Check required packages
    print("\n[2/5] Checking required Python packages...")
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"),
        ("bitsandbytes", "BitsAndBytes"),
        ("piper", "Piper-TTS"),
        ("sounddevice", "SoundDevice"),
        ("soundfile", "SoundFile"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("keyboard", "Keyboard"),
        ("openwakeword", "openWakeWord"),
    ]

    for module, package in required_packages:
        if not check_module(module, package):
            all_good = False

    # Check model directories
    print("\n[3/5] Checking model files...")
    project_root = Path(__file__).parent

    # Check Qwen model
    qwen_path = project_root / "Qwen3-8B"
    if check_file(qwen_path, "Qwen3-8B model directory"):
        # Check for key model files
        config_file = qwen_path / "config.json"
        if not check_file(config_file, "Model config.json"):
            print("   ⚠️  Model may be incomplete. Run the download command:")
            print("   huggingface-cli download Qwen/Qwen2.5-8B-Instruct --local-dir Qwen3-8B")
            all_good = False
    else:
        print("   ⚠️  Download the model with:")
        print("   huggingface-cli download Qwen/Qwen2.5-8B-Instruct --local-dir Qwen3-8B")
        all_good = False

    # Check voice files
    voice_path = project_root / "jarvis-voice"
    check_file(voice_path, "Jarvis voice files")

    # Check audio devices
    print("\n[4/5] Checking audio devices...")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if input_devices:
            print(f"✅ Found {len(input_devices)} input device(s)")
        else:
            print("❌ No input devices found")
            all_good = False
    except Exception as e:
        print(f"❌ Error checking audio devices: {e}")
        all_good = False

    # Check CUDA availability (optional)
    print("\n[5/5] Checking GPU acceleration (optional)...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available (CPU mode - slower but still works)")
    except Exception as e:
        print(f"⚠️  Could not check CUDA: {e}")

    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("✅ All checks passed! You're ready to run Jarvis.")
        print("\nRun: python jarvis/main.py")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nInstall missing packages with: pip install -r requirements.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()
