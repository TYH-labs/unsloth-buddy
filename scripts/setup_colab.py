"""
setup_colab.py — Auto-setup Unsloth on a Google Colab VM.

Run this via colab-mcp's execute_code tool to:
1. Detect the assigned GPU (T4/L4/A100)
2. Install Unsloth
3. Verify all ML packages
4. Print a structured JSON status

Usage (from colab-mcp execute_code):
    exec(open("setup_colab.py").read())
    # or paste directly into execute_code

Output: JSON dict with keys:
    status: "ready" | "error"
    gpu: str (e.g. "Tesla T4")
    vram_gb: float
    cuda_version: str
    python_version: str
    packages: dict of installed package versions
    errors: list of error messages (if any)
"""

import subprocess
import sys
import json


def _run(cmd):
    """Run a shell command and return (ok, stdout)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=300
        )
        return result.returncode == 0, result.stdout.strip()
    except Exception as e:
        return False, str(e)


def detect_gpu():
    """Detect NVIDIA GPU model and VRAM."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            cuda_ver = torch.version.cuda or "unknown"
            return name, round(vram, 1), cuda_ver
    except ImportError:
        pass

    # Fallback: nvidia-smi
    ok, out = _run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits")
    if ok and out:
        parts = out.split(",")
        name = parts[0].strip()
        vram = round(float(parts[1].strip()) / 1024, 1) if len(parts) > 1 else 0.0
        return name, vram, "unknown"

    return "No GPU detected", 0.0, "none"


def install_unsloth():
    """Install Unsloth via pip."""
    ok, out = _run(f"{sys.executable} -m pip install --quiet unsloth 2>&1")
    if not ok:
        # Try the auto-install approach for tricky CUDA combos
        ok2, out2 = _run(
            "wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -"
        )
        if not ok2:
            return False, f"pip install failed: {out}; auto-install failed: {out2}"
    return True, "installed"


def verify_packages():
    """Check that all required ML packages are importable."""
    required = [
        "torch", "transformers", "datasets", "trl", "peft",
        "accelerate", "safetensors", "unsloth"
    ]
    versions = {}
    missing = []
    for pkg in required:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            missing.append(pkg)
    return versions, missing


def main():
    errors = []
    status_dict = {
        "status": "ready",
        "gpu": "",
        "vram_gb": 0.0,
        "cuda_version": "",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "packages": {},
        "errors": [],
    }

    # 1. Detect GPU
    gpu_name, vram, cuda_ver = detect_gpu()
    status_dict["gpu"] = gpu_name
    status_dict["vram_gb"] = vram
    status_dict["cuda_version"] = cuda_ver

    if vram == 0.0:
        errors.append("No GPU detected. Make sure Colab runtime is set to GPU.")

    # 2. Install Unsloth
    ok, msg = install_unsloth()
    if not ok:
        errors.append(msg)

    # 3. Verify packages
    versions, missing = verify_packages()
    status_dict["packages"] = versions
    if missing:
        errors.append(f"Missing packages: {', '.join(missing)}")

    # 4. Final status
    if errors:
        status_dict["status"] = "error"
        status_dict["errors"] = errors

    print(json.dumps(status_dict, indent=2))
    return status_dict


if __name__ == "__main__":
    main()
