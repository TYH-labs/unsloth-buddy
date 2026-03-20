import os
import sys
import time
import argparse
import datetime

try:
    import requests
    import plotext as plt
except ImportError:
    print("Error: Missing required packages for the terminal dashboard.")
    print("Please install them using:")
    print("    pip install plotext requests")
    sys.exit(1)

def get_smoothed_loss(data, alpha=0.1):
    """Calculate Exponential Moving Average (EMA) for noisy loss values."""
    if not data:
        return []
    smoothed = [data[0]]
    for i in range(1, len(data)):
        val = data[i]
        prev = smoothed[-1]
        smoothed.append(alpha * val + (1 - alpha) * prev)
    return smoothed

def fetch_metrics(port=8080):
    try:
        response = requests.get(f"http://localhost:{port}/api/metrics", timeout=2)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

def draw_dashboard(payload, theme="clear"):
    plt.clear_terminal()
    
    # ─── 1. Extract Data ───
    phase = payload.get("phase", "unknown").upper()
    logs = payload.get("logs", [])
    hw = payload.get("hardware", {})
    hp = payload.get("hyperparameters", {})
    meta = payload.get("meta", {})
    eta_sec = payload.get("eta_seconds", 0) or 0
    elapsed_sec = payload.get("elapsed_seconds", 0) or 0

    device = hw.get("device", "Unknown GPU")
    vram = hw.get("peak_vram_mb", 0)
    optimizer = hp.get("optimizer", "Auto")
    bs = hp.get("train_batch_size", "?")
    acc = hp.get("gradient_accumulation", "?")
    peak_lr = hp.get("learning_rate", "?")

    eta_str = str(datetime.timedelta(seconds=int(eta_sec))) if eta_sec > 0 else "Calculating..."
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_sec)))

    step = logs[-1].get("step", 0) if logs else 0
    max_steps = meta.get("max_steps", "?")

    # ─── 2. Print Clean Text Header ───
    # Standard terminal print() is much sharper and cleaner than plotext text subplots.
    print("=" * 80)
    print(f" 🚀 Gaslamp Terminal Dashboard   |   Phase: {phase}")
    print(f" ⏱️  Step: {step} / {max_steps}           |   Elapsed: {elapsed_str}   |   ETA: {eta_str}")
    print("-" * 80)
    print(f" 🖥️  Hardware: {device} ({vram} MB Peak VRAM)")
    print(f" ⚙️  Hyperparams: {optimizer}, batch={bs}x{acc}, initial_lr={peak_lr}")
    print("=" * 80)

    # ─── 3. Draw 1x2 Charts ───
    plt.theme(theme)
    plt.subplots(1, 2)
    # Reduce height by 8 to leave room for the text header printed above
    try:
        tw, th = plt.tw(), plt.th()
        plt.plotsize(tw, max(10, th - 8))
    except Exception:
        pass # Fallback if terminal size detection fails

    # Left: Training Loss
    plt.subplot(1, 1)
    train_steps = []
    train_loss = []
    for log in logs:
        if "loss" in log and "step" in log:
            train_steps.append(log["step"])
            train_loss.append(log["loss"])
            
    if train_loss:
        smoothed_loss = get_smoothed_loss(train_loss, alpha=0.1)
        plt.plot(train_steps, train_loss, label="Raw Loss", color="black", marker="dot")
        plt.plot(train_steps, smoothed_loss, label="EMA Loss", color="blue", marker="braille")
        avg_loss = sum(train_loss) / len(train_loss)
        plt.hline(avg_loss, color="red")
        plt.title("Training Loss")
    else:
        plt.title("Training Loss (Waiting for data...)")

    # Right: Eval Loss (or Learning Rate if no eval data yet)
    plt.subplot(1, 2)
    eval_steps = []
    eval_loss = []
    for log in logs:
        if "eval_loss" in log and "step" in log:
            eval_steps.append(log["step"])
            eval_loss.append(log["eval_loss"])

    if eval_loss:
        plt.plot(eval_steps, eval_loss, label="Eval Loss", color="green", marker="braille")
        plt.title("Evaluation Loss")
    else:
        # Fallback to Learning Rate if no evaluation has happened yet
        lr_steps = []
        lrs = []
        for log in logs:
            if "learning_rate" in log and "step" in log:
                lr_steps.append(log["step"])
                lrs.append(log["learning_rate"])
        if lrs:
            plt.plot(lr_steps, lrs, label="Learning Rate", color="yellow")
            plt.title("Learning Rate Schedule")
        else:
            plt.title("Evaluation (Waiting for data...)")

    # Render layout
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Gaslamp Terminal Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Gaslamp callback port")
    parser.add_argument("--theme", type=str, default="clear", help="Plotext theme: clear, dark, pro, matrix, ubuntu, windows, retro, elegant")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval (seconds)")
    args = parser.parse_args()

    print(f"Connecting to Gaslamp training process on http://localhost:{args.port}...")
    
    try:
        while True:
            payload = fetch_metrics(args.port)
            if payload:
                draw_dashboard(payload, theme=args.theme)
                if payload.get("phase", "").lower() in ["completed", "error"]:
                    print(f"\nTraining finished with state: {payload.get('phase')}")
                    break
            else:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"Waiting for training server on port {args.port}...")
                
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\nExiting terminal dashboard...")
        sys.exit(0)

if __name__ == "__main__":
    main()
