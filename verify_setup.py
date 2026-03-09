#!/usr/bin/env python3
"""
Verify Jarvis AI setup and installation.
Run this script after installation to check if everything is configured correctly.
"""

import importlib.util
import sys
from pathlib import Path


def check_module(name, package=None):
    """Check if a Python module is installed."""
    pkg = package or name
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"[FAIL] {pkg} is NOT installed")
        return False

    print(f"[OK] {pkg} is installed")
    return True


def check_file(path, description):
    """Check if a file or directory exists."""
    if path.exists():
        print(f"[OK] {description} found at: {path}")
        return True

    print(f"[FAIL] {description} NOT found at: {path}")
    return False


def main():
    print("=" * 60)
    print("Jarvis AI Setup Verification")
    print("=" * 60)

    all_good = True

    print("\n[1/5] Checking Python version...")
    py_version = sys.version_info
    if py_version >= (3, 9):
        print(f"[OK] Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print(f"[FAIL] Python {py_version.major}.{py_version.minor}.{py_version.micro} (3.9+ required)")
        all_good = False

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

    print("\n[3/5] Checking model files...")
    project_root = Path(__file__).parent

    qwen_path = project_root / "Qwen3.5-9B"
    llm_found = False

    if qwen_path.exists():
        config_file = qwen_path / "config.json"
        if config_file.exists():
            print("[OK] Qwen3.5-9B model found (recommended)")
            llm_found = True
        else:
            print("[WARN] Qwen3.5-9B directory exists but may be incomplete (missing config.json)")

    if not llm_found:
        potential_llm_dirs = [
            d for d in project_root.iterdir()
            if d.is_dir()
            and (d / "config.json").exists()
            and any(f.suffix in [".safetensors", ".bin"] for f in d.iterdir() if f.is_file())
        ]

        if potential_llm_dirs:
            for llm_dir in potential_llm_dirs:
                print(f"[OK] LLM model found: {llm_dir.name}")
                llm_found = True
            print("   [INFO] Qwen3.5-9B is recommended for best results")

    if not llm_found:
        print("[FAIL] No LLM model found")
        print("   [WARN] Download the recommended model with:")
        print("   huggingface-cli download Qwen/Qwen3.5-9B --local-dir Qwen3.5-9B")
        all_good = False

    try:
        sys.path.insert(0, str(project_root / "jarvis"))
        import config as jarvis_config

        if jarvis_config.MULTIMODAL:
            print("\n   Checking multimodal dependencies...")
            multimodal_packages = [
                ("PIL", "Pillow"),
                ("torchvision", "torchvision"),
                ("qwen_vl_utils", "qwen-vl-utils"),
            ]
            for module, package in multimodal_packages:
                if not check_module(module, package):
                    all_good = False
    except Exception:
        pass

    voice_path = project_root / "jarvis-voice"
    check_file(voice_path, "Jarvis voice files")

    print("\n[4/5] Checking audio devices...")
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        input_devices = [d for d in devices if d["max_input_channels"] > 0]
        if input_devices:
            print(f"[OK] Found {len(input_devices)} input device(s)")
        else:
            print("[FAIL] No input devices found")
            all_good = False
    except Exception as exc:
        print(f"[FAIL] Error checking audio devices: {exc}")
        all_good = False

    print("\n[5/5] Checking GPU acceleration (optional)...")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            torch_version = getattr(torch, "__version__", "unknown")
            if "+cpu" in torch_version:
                print("[WARN] CPU-only PyTorch build detected")
                print("   Install the CUDA build in this venv with:")
                print("   pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
            else:
                print("[WARN] CUDA not available (CPU mode - slower but still works)")
    except Exception as exc:
        print(f"[WARN] Could not check CUDA: {exc}")

    print("\n" + "=" * 60)
    if all_good:
        print("[OK] All checks passed. You're ready to run Jarvis.")
        print("\nRun: python jarvis/main.py")
    else:
        print("[FAIL] Some checks failed. Please fix the issues above.")
        print("\nInstall missing packages with: pip install -r requirements.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
