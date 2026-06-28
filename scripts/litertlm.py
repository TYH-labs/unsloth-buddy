#!/usr/bin/env python3
"""
litertlm.py — Unified LiteRT-LM CLI for unsloth-buddy.

Subcommands:
    install     Install/upgrade litert-lm-api and litert-lm-builder
    bundle      Bundle a TFLite model, tokenizer, and metadata into a .litertlm file
    serve       Start a local OpenAI-compatible API server + open chat WebUI
    chat        Interactive terminal chat session with a .litertlm model
    deploy      Auto-pipeline: bundle → serve

Usage:
    python scripts/litertlm.py install
    python scripts/litertlm.py bundle --tflite model.tflite --tokenizer tokenizer.model --output model.litertlm
    python scripts/litertlm.py serve --model model.litertlm --port 8082
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

# ─── Constants ──────────────────────────────────────────────────────
LITERT_BINS = ["litert-lm", "litert-lm-peek", "litert-lm-builder"]

# ─── Utility ────────────────────────────────────────────────────────

def _find_bin(name: str) -> Optional[str]:
    """Locate a LiteRT-LM binary on PATH."""
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


def _detect_litert() -> dict:
    """Detect installed LiteRT-LM binaries."""
    info = {"installed": False, "version": None, "binaries": {}}
    for name in LITERT_BINS:
        path = _find_bin(name)
        if path:
            info["binaries"][name] = path
            info["installed"] = True
    # Try to get version
    cli = info["binaries"].get("litert-lm")
    if cli:
        try:
            r = subprocess.run([cli, "--version"], capture_output=True, text=True, timeout=5)
            # litert-lm output might contain version number
            lines = (r.stdout + r.stderr).splitlines()
            if lines:
                info["version"] = lines[0].strip()
        except Exception:
            pass
    return info


# ─── Local HTTP Server for serving .litertlm ────────────────────────

class OpenAICompatibleHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Mute standard http logs to avoid output spam
        pass

    def send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            model_id = os.path.basename(self.server.model_path) if hasattr(self.server, 'model_path') else "litert-lm"
            resp = {
                "object": "list",
                "data": [{
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "litert-lm"
                }]
            }
            self.wfile.write(json.dumps(resp).encode('utf-8'))
        elif self.path == "/health":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ready"}).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_cors_headers()
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            req = {}
            try:
                req = json.loads(post_data.decode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Invalid JSON: {e}"}).encode('utf-8'))
                return

            messages = req.get("messages", [])
            stream = req.get("stream", False)
            user_msg = ""
            for m in messages:
                if m.get("role") == "user":
                    user_msg = m.get("content", "")

            # Run inference
            if not hasattr(self.server, 'engine') or self.server.engine is None:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": "LiteRT-LM engine is not initialized."}).encode('utf-8'))
                return

            try:
                if stream:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-cache')
                    self.send_header('Connection', 'keep-alive')
                    self.send_cors_headers()
                    self.end_headers()

                    # Create a new conversation per request to avoid context blending
                    with self.server.engine.create_conversation() as conv:
                        for chunk in conv.send_message_async(user_msg):
                            text = chunk.get("content", [{}])[0].get("text", "")
                            event_data = {
                                "choices": [{
                                    "delta": {"content": text},
                                    "index": 0,
                                    "finish_reason": None
                                }]
                            }
                            self.wfile.write(f"data: {json.dumps(event_data)}\n\n".encode('utf-8'))
                            self.wfile.flush()
                        
                        # Send done signal
                        done_event = {
                            "choices": [{
                                "delta": {},
                                "index": 0,
                                "finish_reason": "stop"
                            }]
                        }
                        self.wfile.write(f"data: {json.dumps(done_event)}\n\ndata: [DONE]\n\n".encode('utf-8'))
                        self.wfile.flush()
                else:
                    # Non-streamed response
                    response_text = ""
                    with self.server.engine.create_conversation() as conv:
                        for chunk in conv.send_message_async(user_msg):
                            response_text += chunk.get("content", [{}])[0].get("text", "")
                    
                    resp = {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": response_text
                            },
                            "finish_reason": "stop",
                            "index": 0
                        }],
                        "object": "chat.completion",
                        "model": os.path.basename(self.server.model_path)
                    }
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps(resp).encode('utf-8'))
            except Exception as e:
                # Handle error
                if not stream:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": f"Inference engine failure: {e}"}).encode('utf-8'))
                else:
                    # In a stream, write error chunk and end stream
                    err_event = {
                        "choices": [{
                            "delta": {"content": f"\n[Backend Error: {e}]"},
                            "index": 0,
                            "finish_reason": "stop"
                        }]
                    }
                    try:
                        self.wfile.write(f"data: {json.dumps(err_event)}\n\ndata: [DONE]\n\n".encode('utf-8'))
                        self.wfile.flush()
                    except Exception:
                        pass
        else:
            self.send_response(404)
            self.send_cors_headers()
            self.end_headers()


# ─── Subcommands ────────────────────────────────────────────────────

def cmd_install(args):
    """Install/upgrade LiteRT-LM framework packages."""
    print("⏳  Checking/Installing LiteRT-LM packages...")
    packages = ["litert-lm-api", "litert-lm-builder"]
    
    # Check if uv is available
    if shutil.which("uv"):
        print("⚡  Found uv package manager. Using 'uv pip install'...")
        cmd = ["uv", "pip", "install", "--upgrade"] + packages
    else:
        print("📦  Using standard 'pip install'...")
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
        
    try:
        subprocess.run(cmd, check=True)
        print("✅  Successfully installed LiteRT-LM packages!")
    except subprocess.CalledProcessError as e:
        print(f"❌  Package installation failed: {e}")
        sys.exit(1)


def cmd_bundle(args):
    """Bundle TFLite flatbuffer, tokenizer, and metadata into a .litertlm file."""
    builder_bin = _require_bin("litert-lm-builder")
    
    if not os.path.isfile(args.tflite):
        print(f"❌  TFLite model file not found: {args.tflite}")
        sys.exit(1)
    if not os.path.isfile(args.tokenizer):
        print(f"❌  Tokenizer file not found: {args.tokenizer}")
        sys.exit(1)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    print(f"📦  Bundling model assets into: {args.output}")
    print(f"    - TFLite Model : {args.tflite} ({_file_size_mb(args.tflite):.1f} MB)")
    print(f"    - Tokenizer    : {args.tokenizer}")
    
    # Construct Builder Command
    # litert-lm-builder system_metadata --str Authors "ODML Team" tflite_model --path <path> --model_type <type> sp_tokenizer --path <path> output --path <path>
    cmd = [
        builder_bin,
        "system_metadata", "--str", "Authors", args.author,
        "tflite_model", "--path", args.tflite, "--model_type", args.model_type,
        "sp_tokenizer", "--path", args.tokenizer,
        "output", "--path", args.output
    ]
    
    try:
        _run(cmd, capture=False)
        print(f"✅  Created .litertlm model: {args.output} ({_file_size_mb(args.output):.1f} MB)")
    except subprocess.CalledProcessError as e:
        print(f"❌  Bundling failed: {e}")
        sys.exit(1)


def cmd_serve(args):
    """Start local HTTP server wrapping litert_lm.Engine + open chat WebUI."""
    if not os.path.isfile(args.model):
        print(f"❌  Model file not found: {args.model}")
        sys.exit(1)

    print(f"🚀  Loading LiteRT-LM Engine with: {args.model}...")

    # Load package dynamically
    try:
        import litert_lm
    except ImportError:
        print("❌  'litert_lm' package not found in Python path.")
        print(f"    Run:  python {__file__} install")
        sys.exit(1)

    # Select backend
    backend_choice = args.backend.lower()
    if backend_choice == "gpu":
        backend = litert_lm.Backend.GPU()
    elif backend_choice == "npu":
        backend = litert_lm.Backend.NPU()
    else:
        backend = litert_lm.Backend.CPU()

    try:
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
    except Exception:
        pass

    try:
        engine = litert_lm.Engine(args.model, backend=backend)
    except Exception as e:
        print(f"❌  Failed to initialize LiteRT-LM Engine: {e}")
        sys.exit(1)

    # Spin up server
    server = HTTPServer(('localhost', args.port), OpenAICompatibleHandler)
    server.engine = engine
    server.model_path = args.model
    # Cache a single conversation just in case, though handler creates new ones
    server.conversation = engine.create_conversation()

    print(f"✅  LiteRT-LM server running locally at http://localhost:{args.port}")
    print(f"    API:    http://localhost:{args.port}/v1/chat/completions")

    # Find Chat WebUI
    chat_ui_path = _find_chat_ui()
    if chat_ui_path and not args.no_open:
        url = f"file://{chat_ui_path}?port={args.port}"
        print(f"    WebUI:  {url}")
        webbrowser.open(url)
    else:
        print(f"    WebUI:  Open templates/chat_ui.html in browser and specify port {args.port}")

    def _sigint_handler(sig, frame):
        print("\n🛑  Stopping server...")
        try:
            server.conversation.close()
        except Exception:
            pass
        server.server_close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    print("\n    Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.conversation.close()
        except Exception:
            pass
        server.server_close()


def cmd_chat(args):
    """Direct terminal chat session with .litertlm model."""
    if not os.path.isfile(args.model):
        print(f"❌  Model file not found: {args.model}")
        sys.exit(1)

    # Load package dynamically
    try:
        import litert_lm
    except ImportError:
        print("❌  'litert_lm' package not found in Python path.")
        print(f"    Run:  python {__file__} install")
        sys.exit(1)

    # Select backend
    backend_choice = args.backend.lower()
    if backend_choice == "gpu":
        backend = litert_lm.Backend.GPU()
    elif backend_choice == "npu":
        backend = litert_lm.Backend.NPU()
    else:
        backend = litert_lm.Backend.CPU()

    try:
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
    except Exception:
        pass

    print(f"⚡  Loading model into LiteRT-LM Engine...")
    try:
        with litert_lm.Engine(args.model, backend=backend) as engine:
            with engine.create_conversation() as conv:
                print("✅  Model loaded successfully!")
                print("💬  Enter chat session. Press Ctrl+C or type 'exit' to quit.\n")
                
                while True:
                    try:
                        user_input = input("\033[92mUser > \033[0m").strip()
                        if not user_input:
                            continue
                        if user_input.lower() in ("exit", "quit"):
                            break
                        
                        print("\033[94mLiteRT-LM > \033[0m", end="", flush=True)
                        for chunk in conv.send_message_async(user_input):
                            text = chunk.get("content", [{}])[0].get("text", "")
                            print(text, end="", flush=True)
                        print("\n")
                    except KeyboardInterrupt:
                        print("\n👋  Exiting chat...")
                        break
                    except Exception as e:
                        print(f"\n❌  Inference error: {e}\n")
    except Exception as e:
        print(f"❌  Failed to run model: {e}")
        sys.exit(1)


def cmd_deploy(args):
    """Auto-pipeline: bundle → serve."""
    if args.tflite.endswith(".litertlm"):
        if not os.path.isfile(args.tflite):
            print(f"❌  LiteRT-LM model file not found: {args.tflite}")
            sys.exit(1)
        print("📌  Model argument is already a .litertlm file. Direct serving...")
        serve_args = argparse.Namespace(
            model=args.tflite,
            port=args.port,
            backend=args.backend,
            no_open=args.no_open
        )
        cmd_serve(serve_args)
        return

    if not os.path.isfile(args.tflite):
        print(f"❌  TFLite model file not found: {args.tflite}")
        sys.exit(1)
    if not args.tokenizer:
        print("❌  --tokenizer is required when bundling a .tflite model.")
        print("    To serve an existing bundle, run: python scripts/litertlm.py serve --model outputs/model.litertlm")
        sys.exit(1)

    # 1. Bundle
    print(f"\n{'━'*60}\n  Step 1/2: Bundling TFLite assets\n{'━'*60}")
    bundle_args = argparse.Namespace(
        tflite=args.tflite,
        tokenizer=args.tokenizer,
        output=args.output,
        author=args.author,
        model_type=args.model_type
    )
    cmd_bundle(bundle_args)

    # 2. Serve
    print(f"\n{'━'*60}\n  Step 2/2: Serving model\n{'━'*60}")
    serve_args = argparse.Namespace(
        model=args.output,
        port=args.port,
        backend=args.backend,
        no_open=args.no_open
    )
    cmd_serve(serve_args)


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
        prog="litertlm",
        description="Unified LiteRT-LM CLI for unsloth-buddy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Quick start:
              %(prog)s install
              %(prog)s deploy --tflite model.tflite --tokenizer tokenizer.model --output model.litertlm
              %(prog)s serve --model model.litertlm
        """),
    )
    sub = p.add_subparsers(dest="command", required=True)

    # install
    sub.add_parser("install", help="Install LiteRT-LM components via pip")

    # bundle
    sb = sub.add_parser("bundle", help="Package TFLite assets into a .litertlm file")
    sb.add_argument("--tflite", required=True, help="Path to input .tflite model file")
    sb.add_argument("--tokenizer", required=True, help="Path to SentencePiece tokenizer file (.model)")
    sb.add_argument("--output", required=True, help="Path to output .litertlm file")
    sb.add_argument("--author", default="ODML Team", help="Model author metadata")
    sb.add_argument("--model-type", default="gemma", help="Model type metadata (e.g. gemma, gemma4)")

    # serve
    ss = sub.add_parser("serve", help="Serve a .litertlm model over local OpenAI API")
    ss.add_argument("--model", required=True, help="Path to .litertlm file")
    ss.add_argument("--port", type=int, default=8082, help="Local HTTP server port")
    ss.add_argument("--backend", default="cpu", choices=["cpu", "gpu", "npu"], help="Hardware acceleration backend")
    ss.add_argument("--no-open", action="store_true", help="Do not auto-open chat UI browser")

    # chat
    sc = sub.add_parser("chat", help="Start interactive terminal chat session")
    sc.add_argument("--model", required=True, help="Path to .litertlm file")
    sc.add_argument("--backend", default="cpu", choices=["cpu", "gpu", "npu"], help="Hardware backend")

    # deploy
    sd = sub.add_parser("deploy", help="Unified bundle and serve pipeline")
    sd.add_argument("--tflite", required=True, help="Path to .tflite model file (or serving .litertlm file)")
    sd.add_argument("--tokenizer", help="Path to tokenizer file (required for bundling)")
    sd.add_argument("--output", default="outputs/model.litertlm", help="Path to packaged .litertlm output")
    sd.add_argument("--port", type=int, default=8082, help="Local HTTP server port")
    sd.add_argument("--backend", default="cpu", choices=["cpu", "gpu", "npu"], help="Hardware backend")
    sd.add_argument("--author", default="ODML Team", help="Model author metadata")
    sd.add_argument("--model-type", default="gemma", help="Model type metadata")
    sd.add_argument("--no-open", action="store_true", help="Do not auto-open WebUI browser")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "install":  cmd_install,
        "bundle":   cmd_bundle,
        "serve":    cmd_serve,
        "chat":     cmd_chat,
        "deploy":   cmd_deploy,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
