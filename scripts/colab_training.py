"""
colab_training.py — Helper utilities for remote Unsloth training on Google Colab.

These functions are used by the AI agent (not the user directly) to manage
the Colab training workflow via colab-mcp's execute_code MCP tool.

Workflow:
    1. setup()     — install Unsloth and verify GPU
    2. upload()    — send dataset/script to Colab kernel
    3. train()     — execute training and stream logs
    4. download()  — retrieve adapters/GGUF back to local
"""

import base64
import json
import os
import textwrap


# ---------------------------------------------------------------------------
# Code generators — these produce Python code strings to run via execute_code
# ---------------------------------------------------------------------------

def generate_setup_code():
    """Return Python code to install Unsloth and verify the environment on Colab."""
    # Read the setup_colab.py script and wrap it as inline code
    setup_script = os.path.join(os.path.dirname(__file__), "setup_colab.py")
    if os.path.exists(setup_script):
        with open(setup_script, "r") as f:
            return f.read() + "\nmain()"
    # Fallback: inline install
    return textwrap.dedent("""\
        import subprocess, sys, json
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "unsloth"])
        import torch
        print(json.dumps({
            "status": "ready",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1) if torch.cuda.is_available() else 0
        }))
    """)


def generate_upload_code(local_path, remote_path="/content/data"):
    """Return Python code that writes a base64-encoded file to the Colab filesystem.

    Best for small files (<10MB). For larger datasets, use HuggingFace Hub download instead.

    Args:
        local_path: Path to the local file to upload.
        remote_path: Destination path on the Colab VM.

    Returns:
        Python code string for execute_code.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    file_size = os.path.getsize(local_path)
    if file_size > 10 * 1024 * 1024:  # 10MB
        raise ValueError(
            f"File too large for direct upload ({file_size / 1024 / 1024:.1f}MB). "
            "Use generate_hf_download_code() for large datasets."
        )

    with open(local_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")

    filename = os.path.basename(local_path)
    return textwrap.dedent(f"""\
        import base64, os
        os.makedirs("{remote_path}", exist_ok=True)
        data = base64.b64decode("{encoded}")
        path = os.path.join("{remote_path}", "{filename}")
        with open(path, "wb") as f:
            f.write(data)
        print(f"Uploaded {{len(data)}} bytes to {{path}}")
    """)


def generate_hf_download_code(dataset_name, split="train", remote_path="/content/data"):
    """Return Python code to download a dataset from HuggingFace Hub on Colab.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. "yahma/alpaca-cleaned").
        split: Dataset split to download.
        remote_path: Where to cache the dataset.

    Returns:
        Python code string for execute_code.
    """
    return textwrap.dedent(f"""\
        from datasets import load_dataset
        ds = load_dataset("{dataset_name}", split="{split}", cache_dir="{remote_path}")
        print(f"Downloaded {{len(ds)}} rows from {dataset_name} ({{split}})")
        print(f"Columns: {{ds.column_names}}")
        print(ds[0])
    """)


def generate_training_code(train_script_content):
    """Return the training script content as-is for execute_code.

    The agent generates train.py using the normal unsloth-buddy Phase 4 logic,
    then passes its content here to run on Colab.

    Args:
        train_script_content: The full Python training script as a string.

    Returns:
        Python code string for execute_code.
    """
    return train_script_content


def generate_download_code(remote_path, encoding="base64"):
    """Return Python code that reads a file from Colab and prints it as base64.

    The agent captures the output and decodes it locally to save the file.

    Args:
        remote_path: Path to the file on Colab VM.
        encoding: "base64" for binary files, "text" for text files.

    Returns:
        Python code string for execute_code.
    """
    if encoding == "base64":
        return textwrap.dedent(f"""\
            import base64, os
            path = "{remote_path}"
            if os.path.isdir(path):
                import tarfile, io
                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                    tar.add(path, arcname=os.path.basename(path))
                encoded = base64.b64encode(buf.getvalue()).decode("ascii")
                print(f"TAR_BASE64:{{encoded}}")
            else:
                with open(path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("ascii")
                print(f"FILE_BASE64:{{encoded}}")
        """)
    else:
        return textwrap.dedent(f"""\
            with open("{remote_path}", "r") as f:
                print(f.read())
        """)


def generate_metrics_poll_code(dashboard_port=8080):
    """Return Python code to fetch metrics from the Gaslamp dashboard running on Colab.

    Used for relaying training progress back to the local dashboard.

    Args:
        dashboard_port: The port the Gaslamp dashboard server runs on.

    Returns:
        Python code string for execute_code.
    """
    return textwrap.dedent(f"""\
        import requests, json
        try:
            r = requests.get("http://localhost:{dashboard_port}/api/metrics", timeout=5)
            print(json.dumps(r.json()))
        except Exception as e:
            print(json.dumps({{"error": str(e)}}))
    """)


def generate_list_outputs_code(output_dir="/content/outputs"):
    """Return Python code to list all output files from training.

    Args:
        output_dir: The training output directory on Colab.

    Returns:
        Python code string for execute_code.
    """
    return textwrap.dedent(f"""\
        import os, json
        files = []
        for root, dirs, filenames in os.walk("{output_dir}"):
            for fn in filenames:
                full = os.path.join(root, fn)
                files.append({{
                    "path": full,
                    "size_mb": round(os.path.getsize(full) / (1024*1024), 2)
                }})
        print(json.dumps(files, indent=2))
    """)


# ---------------------------------------------------------------------------
# Local helpers — used by the agent to decode downloaded files
# ---------------------------------------------------------------------------

def decode_base64_output(output_text, local_path):
    """Decode a base64-encoded file output from Colab and save locally.

    Args:
        output_text: The stdout from execute_code containing FILE_BASE64 or TAR_BASE64.
        local_path: Where to save the decoded file locally.
    """
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    for line in output_text.strip().split("\n"):
        if line.startswith("FILE_BASE64:"):
            data = base64.b64decode(line[len("FILE_BASE64:"):])
            with open(local_path, "wb") as f:
                f.write(data)
            return local_path

        if line.startswith("TAR_BASE64:"):
            import tarfile
            import io
            data = base64.b64decode(line[len("TAR_BASE64:"):])
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                tar.extractall(path=os.path.dirname(local_path) or ".")
            return local_path

    raise ValueError("No base64-encoded content found in output")
