"""
detect_env.py — Stage 2 environment detection.

Run from whichever Python environment you intend to train in:

    # venv / uv venv
    source .venv/bin/activate && python scripts/detect_env.py

    # conda / mamba
    conda activate myenv && python scripts/detect_env.py

    # pyenv
    pyenv shell 3.12.0 && python scripts/detect_env.py

    # poetry
    poetry run python scripts/detect_env.py

    # pipenv
    pipenv run python scripts/detect_env.py

    # system / docker / any other
    python scripts/detect_env.py

Checks that all training dependencies are installed in THIS Python.
Exits non-zero if critical packages are missing.
"""

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_pkg(import_name):
    """Return version string, or None if not importable."""
    try:
        mod = importlib.import_module(import_name)
        return getattr(mod, "__version__", "installed")
    except ImportError:
        return None


def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL,
                                       timeout=8, text=True).strip()
    except Exception:
        return ""


# ── Detect environment type (order matters — most specific first) ────────────
def detect_env_type():
    exe = sys.executable

    # Docker
    if Path("/.dockerenv").exists():
        return "docker", sys.prefix, "isolated"

    # Conda / Mamba
    conda_env  = os.environ.get("CONDA_DEFAULT_ENV", "")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_env and conda_env != "base":
        return "conda", conda_prefix, "isolated"
    if conda_env == "base":
        return "conda-base", conda_prefix, "shared"  # warn: pollutes base

    # Poetry
    if os.environ.get("POETRY_ACTIVE"):
        return "poetry", sys.prefix, "isolated"

    # Pipenv
    if os.environ.get("PIPENV_ACTIVE"):
        return "pipenv", sys.prefix, "isolated"

    # venv / virtualenv / uv venv (sys.prefix differs from base_prefix)
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    if in_venv:
        # Distinguish uv-managed venv by checking for uv.lock in project root
        env_label = "uv-venv" if Path("uv.lock").exists() or shutil.which("uv") else "venv"
        return env_label, sys.prefix, "isolated"

    # pyenv shim (not in a venv, but pyenv version is set)
    if os.environ.get("PYENV_VERSION") or ".pyenv" in exe:
        return "pyenv", sys.prefix, "shared"

    # System Python — not isolated
    return "system", sys.prefix, "shared"


env_type, env_prefix, isolation = detect_env_type()

# ── Python identity ───────────────────────────────────────────────────────────
python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
python_path    = sys.executable

# ── Training backend ──────────────────────────────────────────────────────────
unsloth_ver  = check_pkg("unsloth")
mlx_tune_ver = check_pkg("mlx_tune")
mlx_ver      = check_pkg("mlx")

if unsloth_ver:
    backend = "unsloth"
elif mlx_tune_ver:
    backend = "mlx-tune"
else:
    backend = None

# ── Accelerator ───────────────────────────────────────────────────────────────
torch_ver  = check_pkg("torch")
cuda_avail = False
mps_avail  = False
if torch_ver:
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        mps_avail  = torch.backends.mps.is_available()
    except Exception:
        pass

# ── ML packages ───────────────────────────────────────────────────────────────
packages = {
    "transformers":    check_pkg("transformers"),
    "datasets":        check_pkg("datasets"),
    "trl":             check_pkg("trl"),
    "peft":            check_pkg("peft"),
    "accelerate":      check_pkg("accelerate"),
    "safetensors":     check_pkg("safetensors"),
    "huggingface_hub": check_pkg("huggingface_hub"),
}

# ── HF cache & disk ───────────────────────────────────────────────────────────
hf_cache     = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
hf_cache_ok  = hf_cache.exists()
disk_free_gb = 0
try:
    disk_free_gb = round(shutil.disk_usage(Path.home()).free / 1024**3, 1)
except Exception:
    pass

# ── Install hint — varies by env type ────────────────────────────────────────
_install_hints = {
    "conda":      "conda install -c conda-forge {pkg}  OR  pip install {pkg}",
    "conda-base": "pip install {pkg}  (consider creating a dedicated conda env first)",
    "poetry":     "poetry add {pkg}",
    "pipenv":     "pipenv install {pkg}",
    "uv-venv":    "uv pip install {pkg}",
    "venv":       "pip install {pkg}",
    "pyenv":      "pip install {pkg}",
    "docker":     "pip install {pkg}",
    "system":     "pip install --user {pkg}  (recommend creating an isolated env first)",
}
install_hint = _install_hints.get(env_type, "pip install {pkg}")

# ── Readiness issues ──────────────────────────────────────────────────────────
issues = []

if isolation == "shared" and env_type != "docker":
    issues.append(
        f"Running in a shared/system Python ({env_type}) — package conflicts are possible. "
        f"Consider an isolated env: uv venv .venv --python 3.12 && source .venv/bin/activate"
    )
if not backend:
    issues.append(
        f"No training backend found. Install one:\n"
        f"    Apple Silicon : {install_hint.format(pkg='mlx-tune')}\n"
        f"    Linux/CUDA    : {install_hint.format(pkg='unsloth')}"
    )
if backend == "unsloth" and not cuda_avail:
    issues.append("Unsloth installed but CUDA unavailable — check GPU driver and CUDA version")
if backend == "mlx-tune" and not mlx_ver:
    issues.append(f"mlx-tune installed but mlx core missing — run: {install_hint.format(pkg='mlx-tune')}")
if not packages["datasets"]:
    issues.append(f"datasets not installed — run: {install_hint.format(pkg='datasets')}")
if not hf_cache_ok:
    issues.append(f"HuggingFace cache missing ({hf_cache}) — run: mkdir -p \"{hf_cache}\"")
if disk_free_gb < 10:
    issues.append(f"Low disk space: {disk_free_gb} GB free (model downloads need 5–20 GB)")

# Shared-env isolation is a warning, not a hard block — don't block readiness for it alone
hard_issues = [i for i in issues if "shared/system" not in i]
ready = len(hard_issues) == 0

# ── Report ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STAGE 2 — PYTHON ENV REPORT")
print("=" * 60)
print(f"Env type       : {env_type}  [{isolation}]")
print(f"Env prefix     : {env_prefix}")
print(f"Python         : {python_version} @ {python_path}")
print(f"Install hint   : {install_hint.format(pkg='<package>')}")

print(f"\nTraining backend : {backend or 'NOT FOUND'}")
print(f"  unsloth        : {unsloth_ver or '—'}")
print(f"  mlx-tune       : {mlx_tune_ver or '—'}")
print(f"  mlx (core)     : {mlx_ver or '—'}")

print(f"\nAccelerator:")
print(f"  PyTorch        : {torch_ver or '—'}")
print(f"  CUDA available : {cuda_avail}")
print(f"  MPS available  : {mps_avail}")

print(f"\nML packages:")
for pkg, ver in packages.items():
    status = ver or "MISSING"
    flag   = "" if ver else f"  ← {install_hint.format(pkg=pkg)}"
    print(f"  {pkg:<20}: {status}{flag}")

print(f"\nHF cache        : {'OK' if hf_cache_ok else 'MISSING'}  ({hf_cache})")
print(f"Disk free       : {disk_free_gb} GB")

print("\n" + "=" * 60)
if ready and not issues:
    print("READY FOR TRAINING")
elif ready and issues:
    print("READY FOR TRAINING  (with warnings)")
    for i, issue in enumerate(issues, 1):
        print(f"  warning {i}: {issue}")
else:
    print("NOT READY — fix these issues first:")
    for i, issue in enumerate(hard_issues, 1):
        print(f"  {i}. {issue}")
print("=" * 60)

# ── JSON artifact ─────────────────────────────────────────────────────────────
summary = {
    "env_type": env_type, "isolation": isolation,
    "env_prefix": env_prefix,
    "python_version": python_version, "python_path": python_path,
    "backend": backend,
    "versions": {"unsloth": unsloth_ver, "mlx_tune": mlx_tune_ver, "mlx": mlx_ver,
                 "torch": torch_ver, **packages},
    "cuda_available": cuda_avail, "mps_available": mps_avail,
    "hf_cache": str(hf_cache), "hf_cache_ok": hf_cache_ok,
    "disk_free_gb": disk_free_gb,
    "ready": ready, "issues": issues,
}
out_file = Path("detect_env_result.json")
out_file.write_text(json.dumps(summary, indent=2))
print(f"\nJSON summary written to: {out_file}")

sys.exit(0 if ready else 1)
