#!/usr/bin/env python3
"""
llamacpp.py — Unified llama.cpp CLI for unsloth-buddy.

One script, seven subcommands:
    install     Install llama.cpp (brew on macOS, cmake on Linux)
    quantize    Re-quantize a GGUF to one or more quant levels
    bench       Benchmark inference speed for one or more GGUFs
    ppl         Measure perplexity of a GGUF over a text file
    serve       Start an OpenAI-compatible server + open chat UI
    chat        Interactive terminal chat with a GGUF model
    deploy      Auto-pipeline: quantize → bench → serve → chat UI

Usage:
    python scripts/llamacpp.py deploy \\
        --model outputs/model-f16.gguf \\
        --quant q4_k_m q8_0 --bench --serve
"""

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import webbrowser
from typing import Optional

# ─── Constants ──────────────────────────────────────────────────────
LLAMA_BINS = [
    "llama-cli", "llama-server", "llama-quantize",
    "llama-perplexity", "llama-bench",
]

QUANT_TYPES = [
    "q2_k", "q3_k_m", "q3_k_l", "q4_0", "q4_k_m", "q4_k_s",
    "q5_0", "q5_k_m", "q5_k_s", "q6_k", "q8_0", "f16", "f32",
]

_server_proc = None  # track background server for cleanup

# ─── Utility ────────────────────────────────────────────────────────

def _find_bin(name: str) -> Optional[str]:
    """Locate a llama.cpp binary on PATH."""
    return shutil.which(name)


def _require_bin(name: str) -> str:
    """Return path to binary or exit with helpful error."""
    path = _find_bin(name)
    if not path:
        print(f"❌  '{name}' not found on PATH.")
        print(f"    Run:  python {__file__} install")
        sys.exit(1)
    return path


def _run(cmd: list[str], capture=False, check=True, **kwargs):
    """Run a subprocess, printing the command first."""
    print(f"  ▸ {' '.join(cmd)}")
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True, check=check, **kwargs)
    return subprocess.run(cmd, check=check, **kwargs)


def _file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def _detect_gpu_layers() -> int:
    """Return a sensible -ngl value: 999 for GPU offload, 0 for CPU-only."""
    system = platform.system()
    if system == "Darwin":
        # macOS: Metal is always available on Apple Silicon
        machine = platform.machine()
        if machine == "arm64":
            return 999
    # Check for NVIDIA
    if shutil.which("nvidia-smi"):
        return 999
    return 0


def _detect_llamacpp() -> dict:
    """Detect installed llama.cpp binaries and version."""
    info = {"installed": False, "version": None, "binaries": {}}
    for name in LLAMA_BINS:
        path = _find_bin(name)
        if path:
            info["binaries"][name] = path
            info["installed"] = True
    # Try to get version
    cli = info["binaries"].get("llama-cli")
    if cli:
        try:
            r = subprocess.run([cli, "--version"], capture_output=True, text=True, timeout=5)
            for line in (r.stdout + r.stderr).splitlines():
                if "version" in line.lower() or "build" in line.lower():
                    info["version"] = line.strip()
                    break
        except Exception:
            pass
    return info


# ─── Subcommands ────────────────────────────────────────────────────

def cmd_install(args):
    """Install llama.cpp on the current system."""
    existing = _detect_llamacpp()
    if existing["installed"] and not args.force:
        print(f"✅  llama.cpp already installed: {existing['version'] or 'unknown version'}")
        for name, path in existing["binaries"].items():
            print(f"    {name} → {path}")
        return

    system = platform.system()
    if system == "Darwin":
        print("🍺  Installing via Homebrew…")
        _run(["brew", "install", "llama.cpp"])
    elif system == "Linux":
        # Try snap first, fall back to cmake build
        if shutil.which("snap"):
            print("📦  Installing via snap…")
            _run(["sudo", "snap", "install", "llama-cpp", "--edge"], check=False)
        if not _find_bin("llama-cli"):
            print("🔧  Building from source via cmake…")
            build_dir = "/tmp/llama-cpp-build"
            os.makedirs(build_dir, exist_ok=True)
            _run(["git", "clone", "--depth=1",
                  "https://github.com/ggml-org/llama.cpp.git", build_dir],
                 check=False)
            _run(["cmake", "-B", f"{build_dir}/build", "-S", build_dir,
                  "-DCMAKE_BUILD_TYPE=Release"])
            _run(["cmake", "--build", f"{build_dir}/build", "--config", "Release", "-j"])
            _run(["sudo", "cmake", "--install", f"{build_dir}/build"])
    else:
        print(f"⚠️  Auto-install not supported on {system}.")
        print("    Download from: https://github.com/ggml-org/llama.cpp/releases")
        sys.exit(1)

    # Verify
    info = _detect_llamacpp()
    if info["installed"]:
        print(f"✅  llama.cpp installed: {info['version'] or 'OK'}")
    else:
        print("❌  Installation failed. Try manual install:")
        print("    https://github.com/ggml-org/llama.cpp/releases")
        sys.exit(1)


def cmd_quantize(args):
    """Re-quantize a GGUF file to one or more quant levels."""
    bin_path = _require_bin("llama-quantize")
    input_file = args.input
    if not os.path.isfile(input_file):
        print(f"❌  Input file not found: {input_file}")
        sys.exit(1)

    out_dir = args.output_dir or os.path.dirname(input_file) or "."
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_file))[0]
    # Remove any existing quant suffix for clean naming
    for qt in QUANT_TYPES:
        if base.endswith(f"-{qt}"):
            base = base[: -(len(qt) + 1)]
            break

    results = []
    input_size = _file_size_mb(input_file)

    for qt in args.types:
        out_file = os.path.join(out_dir, f"{base}-{qt}.gguf")
        print(f"\n{'─'*60}")
        print(f"  Quantizing → {qt}")
        print(f"{'─'*60}")
        try:
            _run([bin_path, input_file, out_file, qt])
            out_size = _file_size_mb(out_file)
            ratio = out_size / input_size * 100 if input_size > 0 else 0
            results.append({
                "type": qt, "file": out_file,
                "size_mb": round(out_size, 1),
                "ratio_pct": round(ratio, 1),
            })
            print(f"  ✅  {qt}: {out_size:.1f} MB ({ratio:.0f}% of original)")
        except subprocess.CalledProcessError as e:
            print(f"  ❌  {qt} failed: {e}")

    # Summary table
    if results:
        print(f"\n{'═'*60}")
        print(f"  {'Type':<12} {'Size':>10} {'Ratio':>8}   Path")
        print(f"{'─'*60}")
        for r in results:
            print(f"  {r['type']:<12} {r['size_mb']:>8.1f} MB {r['ratio_pct']:>6.1f}%   {r['file']}")
        print(f"{'═'*60}")
    return results


def cmd_bench(args):
    """Benchmark inference speed for one or more GGUF files."""
    bin_path = _require_bin("llama-bench")
    results = []

    for model in args.models:
        if not os.path.isfile(model):
            print(f"  ⚠️  Skipping {model}: file not found")
            continue
        print(f"\n  Benchmarking: {os.path.basename(model)}")
        ngl = args.gpu_layers if args.gpu_layers is not None else _detect_gpu_layers()
        cmd = [
            bin_path,
            "-m", model,
            "-ngl", str(ngl),
            "-p", str(args.prompt_tokens),
            "-n", str(args.gen_tokens),
        ]
        try:
            r = _run(cmd, capture=True)
            results.append({"model": os.path.basename(model), "output": r.stdout})
            print(r.stdout)
        except subprocess.CalledProcessError as e:
            print(f"  ❌  Benchmark failed: {e}")

    return results


def cmd_ppl(args):
    """Measure perplexity of a GGUF model over a text file."""
    bin_path = _require_bin("llama-perplexity")
    if not os.path.isfile(args.model):
        print(f"❌  Model not found: {args.model}")
        sys.exit(1)
    if not os.path.isfile(args.file):
        print(f"❌  Text file not found: {args.file}")
        sys.exit(1)

    ngl = args.gpu_layers if args.gpu_layers is not None else _detect_gpu_layers()
    cmd = [
        bin_path,
        "-m", args.model,
        "-f", args.file,
        "-c", str(args.ctx_size),
        "-ngl", str(ngl),
    ]
    _run(cmd)


def cmd_serve(args):
    """Start llama-server and optionally open the chat UI."""
    global _server_proc
    bin_path = _require_bin("llama-server")

    if args.stop:
        # Kill any running llama-server
        _run(["pkill", "-f", "llama-server"], check=False)
        print("🛑  Stopped llama-server")
        return

    if not args.model:
        print("❌  --model is required to start the server.")
        print(f"    Run:  python {__file__} serve --model path/to/model.gguf")
        sys.exit(1)

    if not os.path.isfile(args.model):
        print(f"❌  Model not found: {args.model}")
        sys.exit(1)

    ngl = args.gpu_layers if args.gpu_layers is not None else _detect_gpu_layers()
    cmd = [
        bin_path,
        "-m", args.model,
        "--port", str(args.port),
        "-c", str(args.ctx_size),
        "-ngl", str(ngl),
    ]

    print(f"\n🚀  Starting llama-server on port {args.port}…")
    _server_proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    # Wait for ready
    ready = False
    for _ in range(60):
        time.sleep(1)
        try:
            import urllib.request
            resp = urllib.request.urlopen(f"http://localhost:{args.port}/health", timeout=2)
            if resp.status == 200:
                ready = True
                break
        except Exception:
            if _server_proc.poll() is not None:
                print("❌  llama-server exited unexpectedly")
                sys.exit(1)

    if not ready:
        print("❌  llama-server did not become ready within 60s")
        sys.exit(1)

    print(f"✅  Server ready!")
    print(f"    API:    http://localhost:{args.port}/v1/chat/completions")
    print(f"    WebUI:  http://localhost:{args.port}/")

    # Open chat UI
    if not args.no_open:
        chat_html = _find_chat_ui()
        if chat_html:
            url = f"file://{os.path.abspath(chat_html)}?port={args.port}"
            print(f"    Chat:   {url}")
            webbrowser.open(url)
        else:
            # Fallback: open the built-in llama.cpp WebUI
            webbrowser.open(f"http://localhost:{args.port}/")

    print(f"\n    Press Ctrl+C to stop the server.")

    # Keep running and forward output
    def _handle_sigint(sig, frame):
        if _server_proc:
            _server_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        for line in _server_proc.stdout:
            sys.stdout.write(f"    [server] {line}")
    except KeyboardInterrupt:
        pass
    finally:
        if _server_proc:
            _server_proc.terminate()


def cmd_chat(args):
    """Interactive terminal chat with a GGUF model."""
    bin_path = _require_bin("llama-cli")
    if not os.path.isfile(args.model):
        print(f"❌  Model not found: {args.model}")
        sys.exit(1)

    ngl = args.gpu_layers if args.gpu_layers is not None else _detect_gpu_layers()
    cmd = [
        bin_path,
        "-m", args.model,
        "-cnv",
        "-c", str(args.ctx_size),
        "-ngl", str(ngl),
    ]
    print(f"💬  Starting chat (Ctrl+C to exit)…\n")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass


def cmd_deploy(args):
    """Auto-pipeline: quantize → benchmark → serve → open chat UI."""
    if not os.path.isfile(args.model):
        print(f"❌  Model not found: {args.model}")
        sys.exit(1)

    gguf_files = []

    # ── Step 1: Quantize ──
    if args.quant:
        print(f"\n{'━'*60}")
        print(f"  Step 1/3: Quantizing to {', '.join(args.quant)}")
        print(f"{'━'*60}")
        quant_args = argparse.Namespace(
            input=args.model,
            types=args.quant,
            output_dir=args.output_dir or os.path.dirname(args.model) or ".",
        )
        results = cmd_quantize(quant_args)
        gguf_files = [r["file"] for r in (results or [])]
    else:
        gguf_files = [args.model]

    if not gguf_files:
        print("❌  No GGUF files produced. Aborting.")
        sys.exit(1)

    # ── Step 2: Benchmark ──
    if args.bench and gguf_files:
        print(f"\n{'━'*60}")
        print(f"  Step 2/3: Benchmarking {len(gguf_files)} model(s)")
        print(f"{'━'*60}")
        bench_args = argparse.Namespace(
            models=gguf_files,
            prompt_tokens=512,
            gen_tokens=128,
            gpu_layers=None,
        )
        cmd_bench(bench_args)

    # ── Step 3: Serve ──
    best_model = gguf_files[0]  # TODO: auto-pick based on bench results
    if len(gguf_files) > 1:
        # Pick smallest by file size as a reasonable default
        best_model = min(gguf_files, key=lambda f: os.path.getsize(f))
        print(f"\n  📌  Auto-selected: {os.path.basename(best_model)} "
              f"({_file_size_mb(best_model):.1f} MB)")

    if args.serve:
        print(f"\n{'━'*60}")
        print(f"  Step 3/3: Starting server with {os.path.basename(best_model)}")
        print(f"{'━'*60}")
        serve_args = argparse.Namespace(
            model=best_model,
            port=args.port,
            ctx_size=args.ctx_size,
            gpu_layers=None,
            no_open=args.no_open,
            stop=False,
        )
        cmd_serve(serve_args)
    else:
        print(f"\n✅  Deploy complete! To serve:")
        print(f"    python {__file__} serve --model {best_model}")


# ─── Chat UI Finder ─────────────────────────────────────────────────

def _find_chat_ui() -> Optional[str]:
    """Locate templates/chat_ui.html relative to this script or cwd."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "templates", "chat_ui.html"),
        os.path.join(os.getcwd(), "templates", "chat_ui.html"),
        os.path.join(os.path.dirname(__file__), "templates", "chat_ui.html"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


# ─── CLI Parser ─────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="llamacpp",
        description="Unified llama.cpp CLI for unsloth-buddy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Quick start:
              %(prog)s install
              %(prog)s deploy --model model-f16.gguf --quant q4_k_m --bench --serve
        """),
    )
    sub = p.add_subparsers(dest="command", required=True)

    # install
    si = sub.add_parser("install", help="Install llama.cpp")
    si.add_argument("--force", action="store_true", help="Re-install even if present")

    # quantize
    sq = sub.add_parser("quantize", help="Re-quantize a GGUF")
    sq.add_argument("--input", required=True, help="Path to input GGUF")
    sq.add_argument("--types", nargs="+", default=["q4_k_m"],
                    choices=QUANT_TYPES, help="Quant type(s)")
    sq.add_argument("--output-dir", help="Output directory (default: same as input)")

    # bench
    sb = sub.add_parser("bench", help="Benchmark inference speed")
    sb.add_argument("--models", nargs="+", required=True, help="GGUF file(s)")
    sb.add_argument("--prompt-tokens", type=int, default=512)
    sb.add_argument("--gen-tokens", type=int, default=128)
    sb.add_argument("--gpu-layers", type=int, default=None)

    # ppl
    sp = sub.add_parser("ppl", help="Measure perplexity")
    sp.add_argument("--model", required=True)
    sp.add_argument("--file", required=True, help="Text file for PPL eval")
    sp.add_argument("--ctx-size", type=int, default=2048)
    sp.add_argument("--gpu-layers", type=int, default=None)

    # serve
    ss = sub.add_parser("serve", help="Start OpenAI-compatible server + chat UI")
    ss.add_argument("--model", help="GGUF model path")
    ss.add_argument("--port", type=int, default=8081)
    ss.add_argument("--ctx-size", type=int, default=4096)
    ss.add_argument("--gpu-layers", type=int, default=None)
    ss.add_argument("--no-open", action="store_true", help="Don't open browser")
    ss.add_argument("--stop", action="store_true", help="Stop running server")

    # chat
    sc = sub.add_parser("chat", help="Interactive terminal chat")
    sc.add_argument("--model", required=True)
    sc.add_argument("--ctx-size", type=int, default=4096)
    sc.add_argument("--gpu-layers", type=int, default=None)

    # deploy
    sd = sub.add_parser("deploy", help="Auto-pipeline: quantize → bench → serve")
    sd.add_argument("--model", required=True, help="Input GGUF (typically f16)")
    sd.add_argument("--quant", nargs="*", default=None,
                    help=f"Quant type(s). Choices: {', '.join(QUANT_TYPES)}")
    sd.add_argument("--bench", action="store_true", help="Run speed benchmark")
    sd.add_argument("--serve", action="store_true", help="Start server after")
    sd.add_argument("--port", type=int, default=8081)
    sd.add_argument("--ctx-size", type=int, default=4096)
    sd.add_argument("--output-dir", help="Output dir for quantized files")
    sd.add_argument("--no-open", action="store_true", help="Don't open browser")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "install":  cmd_install,
        "quantize": cmd_quantize,
        "bench":    cmd_bench,
        "ppl":      cmd_ppl,
        "serve":    cmd_serve,
        "chat":     cmd_chat,
        "deploy":   cmd_deploy,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
