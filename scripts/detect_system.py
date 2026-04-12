"""
detect_system.py — Stage 1 environment detection (stdlib only, no pip packages needed).

Run with ANY Python before creating a venv:
    python3 scripts/detect_system.py

Outputs a structured report + JSON summary that the skill uses to decide:
  - Which training backend to install (unsloth / mlx-tune / docker)
  - Which Python version to pin for the venv
  - Which CUDA wheel variant to use
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, timeout=8):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL,
                                      timeout=timeout, text=True)
        return out.strip()
    except Exception:
        return ""


# ── OS & Architecture ────────────────────────────────────────────────────────
os_name   = platform.system()           # Darwin | Linux | Windows
arch      = platform.machine()          # arm64 | x86_64 | AMD64
uname_str = run("uname -a") or platform.version()

is_mac     = os_name == "Darwin"
is_linux   = os_name == "Linux"
is_windows = os_name == "Windows"
is_arm64   = arch in ("arm64", "aarch64")
is_apple_silicon = is_mac and is_arm64

# ── GPU / Accelerator ────────────────────────────────────────────────────────
nvidia_smi  = run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
cuda_version = ""
gpu_name     = ""
gpu_vram_gb  = 0

if nvidia_smi:
    parts = nvidia_smi.split(",")
    if len(parts) >= 2:
        gpu_name    = parts[0].strip()
        vram_str    = parts[1].strip()          # e.g. "40960 MiB"
        gpu_vram_gb = round(int(vram_str.split()[0]) / 1024) if vram_str.split()[0].isdigit() else 0
    cuda_from_smi = run("nvidia-smi | grep 'CUDA Version'")
    if cuda_from_smi:
        import re
        m = re.search(r"CUDA Version:\s*([\d.]+)", cuda_from_smi)
        cuda_version = m.group(1) if m else ""

apple_chip = ""
unified_memory_gb = 0
if is_apple_silicon:
    chip_line = run("system_profiler SPDisplaysDataType | grep 'Chipset Model'")
    apple_chip = chip_line.split(":")[-1].strip() if chip_line else ""
    mem_bytes  = run("sysctl -n hw.memsize")
    unified_memory_gb = round(int(mem_bytes) / 1024**3) if mem_bytes.isdigit() else 0

# ── Python versions available on system ─────────────────────────────────────
def check_python(cmd):
    v = run(f"{cmd} --version 2>&1")
    return v.replace("Python ", "").strip() if "Python" in v else ""

py_versions = {}
for candidate in ["python3", "python3.12", "python3.11", "python3.10", "python3.9", "python"]:
    v = check_python(candidate)
    if v:
        py_versions[candidate] = v

current_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
current_python_path = sys.executable

# ── Package managers ─────────────────────────────────────────────────────────
pkg_managers = {
    "uv":    shutil.which("uv"),
    "conda": shutil.which("conda"),
    "pip3":  shutil.which("pip3"),
    "brew":  shutil.which("brew"),
    "docker":shutil.which("docker"),
}

# ── Existing environments in project dir ─────────────────────────────────────
cwd = Path.cwd()
existing_envs = []
for candidate in [".venv", "venv", "env"]:
    p = cwd / candidate
    if (p / "bin" / "python").exists() or (p / "Scripts" / "python.exe").exists():
        existing_envs.append(str(p))

conda_envs = run("conda env list 2>/dev/null | grep -v '#'").splitlines() if pkg_managers["conda"] else []

# HF cache
hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
hf_cache_exists = hf_cache.exists()

# ── llama.cpp detection ──────────────────────────────────────────────────────
llamacpp_bins = {}
for _bin in ["llama-cli", "llama-server", "llama-quantize", "llama-perplexity", "llama-bench"]:
    _path = shutil.which(_bin)
    if _path:
        llamacpp_bins[_bin] = _path
llamacpp_installed = len(llamacpp_bins) > 0
llamacpp_version = ""
if "llama-cli" in llamacpp_bins:
    _ver_out = run(f"{llamacpp_bins['llama-cli']} --version 2>&1")
    if _ver_out:
        llamacpp_version = _ver_out.splitlines()[0].strip()

# ── Decision logic ────────────────────────────────────────────────────────────
if is_apple_silicon:
    install_path = "C"   # mlx-tune
    recommended_python = "3.12"  # mlx requires <=3.12
elif is_linux and nvidia_smi:
    install_path = "A"   # standard unsloth (or B for advanced)
    recommended_python = "3.11"
elif is_windows:
    install_path = "D_windows"
    recommended_python = "3.12"
elif pkg_managers["docker"]:
    install_path = "D_docker"
    recommended_python = "3.11"
else:
    install_path = "unknown"
    recommended_python = "3.11"

# ── Report ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STAGE 1 — SYSTEM ENVIRONMENT REPORT")
print("=" * 60)
print(f"OS           : {os_name} ({arch})")
print(f"Apple Silicon: {is_apple_silicon}" + (f" — {apple_chip}, {unified_memory_gb}GB unified" if is_apple_silicon else ""))
print(f"NVIDIA GPU   : {gpu_name or 'none'}" + (f" ({gpu_vram_gb}GB VRAM, CUDA {cuda_version})" if gpu_name else ""))
print(f"\nCurrent Python : {current_python} @ {current_python_path}")
print(f"Available      : {py_versions}")
print(f"\nPackage managers: { {k: bool(v) for k, v in pkg_managers.items()} }")
print(f"Existing venvs  : {existing_envs or 'none'}")
print(f"HF cache exists : {hf_cache_exists} ({hf_cache})")
if llamacpp_installed:
    print(f"llama.cpp       : {llamacpp_version or 'installed'} ({len(llamacpp_bins)} binaries)")
else:
    print(f"llama.cpp       : not installed (optional — run: python {__file__.replace('detect_system.py', 'llamacpp.py')} install)")
print(f"\n→ Recommended install path : {install_path}")
print(f"→ Recommended Python       : {recommended_python}")

if install_path == "C":
    needs_new_venv = not existing_envs
    active_python_ok = sys.version_info <= (3, 12)
    print(f"\n  mlx-tune requires Python ≤ 3.12")
    print(f"  Current Python ({current_python}) OK: {active_python_ok}")
    print(f"  Need new venv: {needs_new_venv}")
    if not hf_cache_exists:
        print(f"  WARNING: HF cache missing — run: mkdir -p ~/.cache/huggingface/hub")

print("\n" + "=" * 60)
print("Run Stage 2 after activating your venv:")
print("  source .venv/bin/activate")
print("  python scripts/detect_env.py")
print("=" * 60)

# JSON for programmatic use
summary = {
    "os": os_name, "arch": arch,
    "is_apple_silicon": is_apple_silicon,
    "apple_chip": apple_chip,
    "unified_memory_gb": unified_memory_gb,
    "gpu_name": gpu_name,
    "gpu_vram_gb": gpu_vram_gb,
    "cuda_version": cuda_version,
    "current_python": current_python,
    "available_pythons": py_versions,
    "pkg_managers": {k: bool(v) for k, v in pkg_managers.items()},
    "existing_envs": existing_envs,
    "hf_cache_exists": hf_cache_exists,
    "llama_cpp": {
        "installed": llamacpp_installed,
        "version": llamacpp_version,
        "binaries": llamacpp_bins,
    },
    "install_path": install_path,
    "recommended_python": recommended_python,
}
out_file = Path("detect_system_result.json")
out_file.write_text(json.dumps(summary, indent=2))
print(f"\nJSON summary written to: {out_file}")
