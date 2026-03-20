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

def draw_dashboard(payload):
    plt.clear_terminal()
    
    # Extract data
    phase = payload.get("phase", "unknown")
    logs = payload.get("logs", [])
    hw = payload.get("hardware", {})
    hp = payload.get("hyperparameters", {})
    meta = payload.get("meta", {})
    eta_sec = payload.get("eta_seconds", 0) or 0
    elapsed_sec = payload.get("elapsed_seconds", 0) or 0

    # Layout configuration
    plt.theme("clear")
    plt.subplots(2, 2)
    plt.plotsize(plt.tw(), plt.th())
    
    # ─── Top Left: Training Loss ───
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
        plt.hline(avg_loss, color="red", label="Avg")
        plt.title("Training Loss")
    else:
        plt.title("Training Loss (Waiting for data...)")
        
    # ─── Top Right: Learning Rate ───
    plt.subplot(1, 2)
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
        plt.title("Learning Rate (Waiting for data...)")

    # ─── Bottom Left: Quick Stats ───
    plt.subplot(2, 1)
    eta_str = str(datetime.timedelta(seconds=int(eta_sec))) if eta_sec > 0 else "Calculating..."
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_sec)))
    
    device = hw.get("device", "Unknown")
    vram = hw.get("peak_vram_mb", 0)
    
    stats_text = (
        f"--- Gaslamp Terminal Dashboard ---\n\n"
        f"Phase: {phase.upper()}\n"
        f"Step: {train_steps[-1] if train_steps else 0} / {meta.get('max_steps', '?')}\n"
        f"Elapsed Time: {elapsed_str}\n"
        f"Time Remaining: {eta_str}\n\n"
        f"--- Hardware ---\n"
        f"Device: {device}\n"
        f"Peak VRAM: {vram} MB\n\n"
        f"--- Hyperparameters ---\n"
        f"Optimizer: {hp.get('optimizer', 'Unknown')}\n"
        f"Batch Size: {hp.get('train_batch_size', '?')} (Acc: {hp.get('gradient_accumulation', '?')})\n"
        f"Peak LR: {hp.get('learning_rate', '?')}\n"
    )
    # Since plotext doesn't strictly have a "text box" in subplots easily,
    # we can use a scatter plot with no markers and a title, then just print 
    # the text, but a cleaner way is just an empty plot with title, 
    # relying on the terminal printing after layout.
    # Alternatively, use a bar chart for categorical data if we want.
    plt.title("Overview")
    plt.xticks([])
    plt.yticks([])
    plt.text(stats_text, 0, 0, alignment="center") # Simple text center hack
    
    # ─── Bottom Right: Evaluation Metrics ───
    plt.subplot(2, 2)
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
        plt.title("Evaluation Loss (Waiting for eval...)")

    # Render layout
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Gaslamp Terminal Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Gaslamp callback port")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval (seconds)")
    args = parser.parse_args()

    print(f"Connecting to Gaslamp training process on http://localhost:{args.port}...")
    
    try:
        while True:
            payload = fetch_metrics(args.port)
            if payload:
                draw_dashboard(payload)
                if payload.get("phase") in ["completed", "error"]:
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
