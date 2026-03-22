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


def get_smoothed(data, alpha=0.1):
    """Exponential Moving Average smoothing."""
    if not data:
        return []
    out = [data[0]]
    for v in data[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def fetch_metrics(port=8080):
    try:
        r = requests.get(f"http://localhost:{port}/api/metrics", timeout=2)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _c(col, is_tty):
    """Return color only when rendering in a real TTY."""
    return col if is_tty else "default"


def draw_dashboard(payload, theme="clear", once=False, mode=None, alpha=0.1):
    if not once:
        plt.clear_terminal()

    is_tty = sys.stdout.isatty()

    # ─── 1. Extract Data ───────────────────────────────────────────
    phase        = payload.get("phase", "unknown").upper()
    logs         = payload.get("logs", [])
    hw           = payload.get("hardware", {})
    hp           = payload.get("hyperparameters", {})
    meta         = payload.get("meta", {})
    eta_sec      = payload.get("eta_seconds", 0) or 0
    elapsed_sec  = payload.get("elapsed_seconds", 0) or 0
    train_rt     = payload.get("train_runtime_seconds")

    # Task type: prefer CLI override, fall back to payload, then "sft"
    task_type = (mode or meta.get("task_type", "sft") or "sft").lower()

    device        = hw.get("device", "Unknown GPU")
    vram          = hw.get("peak_vram_mb", 0)
    cpu_ram       = hw.get("cpu_ram_mb", None)
    total_vram    = hw.get("total_vram_mb", 0)
    baseline_vram = hw.get("baseline_vram_mb", 0)
    lora_vram     = hw.get("lora_vram_mb", 0)
    vram_pct      = hw.get("vram_pct", 0)
    lora_pct      = hw.get("lora_vram_pct", 0)
    optimizer = hp.get("optimizer", "Auto")
    bs       = hp.get("train_batch_size", "?")
    acc      = hp.get("gradient_accumulation", "?")
    peak_lr  = hp.get("learning_rate", "?")

    eta_str     = str(datetime.timedelta(seconds=int(eta_sec))) if eta_sec > 0 else "Calculating..."
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_sec)))

    step         = logs[-1].get("step", 0) if logs else 0
    max_steps    = meta.get("max_steps", "?")
    total_epochs = meta.get("total_epochs", 1) or 1
    cur_epoch    = meta.get("current_epoch", 0) or 0

    # Collect per-step series
    train_steps, train_loss = [], []
    eval_steps,  eval_loss  = [], []
    lr_steps,    lrs        = [], []
    gn_steps,    grad_norms = [], []
    tps_steps,   tps_vals   = [], []
    rew_steps,   rewards,   rew_std  = [], [], []
    kl_steps,    kl_vals    = [], []
    cho_steps,   cho_vals   = [], []
    rej_steps,   rej_vals   = [], []

    for log in logs:
        s = log.get("step")
        if s is None:
            continue
        if "loss" in log:
            train_steps.append(s); train_loss.append(log["loss"])
        if "eval_loss" in log:
            eval_steps.append(s);  eval_loss.append(log["eval_loss"])
        if "learning_rate" in log:
            lr_steps.append(s);    lrs.append(log["learning_rate"])
        if "grad_norm" in log:
            gn_steps.append(s);    grad_norms.append(log["grad_norm"])
        if "tokens_per_sec" in log:
            tps_steps.append(s);   tps_vals.append(log["tokens_per_sec"])
        # GRPO
        if "reward" in log:
            rew_steps.append(s)
            rewards.append(log["reward"])
            rew_std.append(log.get("reward_std", 0))
        # KL
        kl = log.get("kl_divergence") or log.get("kl")
        if kl is not None:
            kl_steps.append(s); kl_vals.append(kl)
        # DPO chosen/rejected
        if "rewards_chosen" in log:
            cho_steps.append(s); cho_vals.append(log["rewards_chosen"])
        if "rewards_rejected" in log:
            rej_steps.append(s); rej_vals.append(log["rewards_rejected"])

    # ─── 2. Print Text Header ──────────────────────────────────────
    expand_hint = "  ↕ ctrl+o to expand" if (once and not is_tty) else ""
    print("=" * 80)
    print(f" 🚀 Gaslamp Terminal Dashboard [{task_type.upper()}]{expand_hint}   |   Phase: {phase}")
    print(f" ⏱️  Step: {step} / {max_steps}   |   Elapsed: {elapsed_str}   |   ETA: {eta_str}")
    print(f" 🔄 Epoch: {cur_epoch:.2f} / {total_epochs}")
    print("-" * 80)
    hw_line = f" 🖥️  Hardware: {device} ({vram} MB Peak VRAM)"
    if cpu_ram:
        hw_line += f"  |  CPU RAM: {cpu_ram} MB"
    print(hw_line)
    # Memory breakdown (mirrors unsloth-studio Colab cell 12)
    if total_vram:
        print(f" 📦 Total GPU: {total_vram} MB   |   Model baseline: {baseline_vram} MB   |   LoRA delta: {lora_vram} MB")
        print(f" 📊 VRAM usage: {vram_pct}%   |   LoRA training: {lora_pct}%")
    print(f" ⚙️  Hyperparams: {optimizer}, batch={bs}×{acc}, initial_lr={peak_lr}")

    # Extra metric row — grad norm and tokens/sec (when available)
    extras = []
    if grad_norms:
        extras.append(f"grad_norm={grad_norms[-1]:.4f}")
    if tps_vals:
        extras.append(f"tokens/sec={tps_vals[-1]:.1f}")
    if extras:
        print(f" 📊 Live: {' | '.join(extras)}")

    print("=" * 80)

    # Completed summary banner (mirrors unsloth-studio Colab cell 12)
    if phase == "COMPLETED" and (total_vram or train_rt):
        rt_str = str(datetime.timedelta(seconds=int(train_rt or elapsed_sec)))
        print()
        print("━" * 80)
        print(" ✅  Training Complete — Memory & Time Summary")
        print("━" * 80)
        if train_rt:
            print(f"   Training time:              {rt_str} ({round((train_rt or 0)/60, 2)} minutes)")
        if total_vram:
            print(f"   Total GPU memory:            {total_vram} MB ({round(total_vram/1024,2)} GB)")
            print(f"   Peak reserved memory:        {vram} MB ({round(vram/1024,2)} GB) — {vram_pct}% of max")
            print(f"   Model load baseline:         {baseline_vram} MB ({round(baseline_vram/1024,2)} GB)")
            print(f"   LoRA training overhead:      {lora_vram} MB ({round(lora_vram/1024,2)} GB) — {lora_pct}% of max")
        print("━" * 80)
        print()

    # ─── 3. Build plotext Charts ────────────────────────────────────
    plt.theme(theme if is_tty else "clear")

    use_2x2 = task_type in ("dpo", "grpo") and (
        (task_type == "dpo" and (cho_vals or rej_vals or kl_vals)) or
        (task_type == "grpo" and (rewards or kl_vals))
    )

    if use_2x2:
        plt.subplots(2, 2)
    else:
        plt.subplots(1, 2)

    try:
        tw, th = plt.tw(), plt.th()
        chart_h = 12 if (once and not is_tty) else max(10, th - 10)
        cols = 2
        rows = 2 if use_2x2 else 1
        plt.plotsize(min(tw or 80, 80), chart_h * rows)
    except Exception:
        pass

    # ── Chart 1: Training Loss (always) ──
    plt.subplot(1, 1)
    if train_loss:
        smoothed = get_smoothed(train_loss, alpha=alpha)
        plt.plot(train_steps, train_loss,  label="Raw Loss", color=_c("black", is_tty), marker="dot")
        plt.plot(train_steps, smoothed,    label=f"EMA(α={alpha})", color=_c("blue", is_tty), marker="braille")
        plt.hline(sum(train_loss) / len(train_loss), color=_c("red", is_tty))
        plt.title("Training Loss")
    else:
        plt.title("Training Loss (waiting...)")

    # ── Chart 2: Eval Loss → LR (always) ──
    plt.subplot(1, 2)
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label="Eval Loss", color=_c("green", is_tty), marker="braille")
        plt.title("Evaluation Loss")
    elif lrs:
        plt.plot(lr_steps, lrs, label="Learning Rate", color=_c("yellow", is_tty))
        plt.title("Learning Rate Schedule")
    else:
        plt.title("Eval / LR (waiting...)")

    # ── Charts 3 & 4: task-specific (DPO or GRPO) ──
    if use_2x2:
        plt.subplot(2, 1)
        if task_type == "grpo":
            if rewards:
                hi = [r + s for r, s in zip(rewards, rew_std)]
                lo = [r - s for r, s in zip(rewards, rew_std)]
                plt.plot(rew_steps, hi,      label="+1σ", color=_c("cyan", is_tty))
                plt.plot(rew_steps, rewards, label="Reward", color=_c("blue", is_tty), marker="braille")
                plt.plot(rew_steps, lo,      label="-1σ", color=_c("cyan", is_tty))
                plt.title("GRPO Reward (± std)")
            else:
                plt.title("Reward (waiting...)")
        elif task_type == "dpo":
            if cho_vals or rej_vals:
                if cho_vals:
                    plt.plot(cho_steps, cho_vals, label="Chosen", color=_c("green", is_tty), marker="braille")
                if rej_vals:
                    plt.plot(rej_steps, rej_vals, label="Rejected", color=_c("red", is_tty), marker="braille")
                plt.title("DPO Chosen vs Rejected Reward")
            else:
                plt.title("DPO Rewards (waiting...)")

        plt.subplot(2, 2)
        if kl_vals:
            plt.plot(kl_steps, kl_vals, label="KL Divergence", color=_c("magenta", is_tty), marker="braille")
            plt.title("KL Divergence")
        else:
            plt.title("KL Divergence (waiting...)")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Gaslamp Terminal Dashboard")
    parser.add_argument("--port",     type=int,   default=8080,    help="Gaslamp callback HTTP port")
    parser.add_argument("--theme",    type=str,   default="clear", help="Plotext theme: clear, dark, pro, matrix, ubuntu, windows, retro, elegant")
    parser.add_argument("--interval", type=float, default=2.0,     help="Refresh interval in seconds")
    parser.add_argument("--once",     action="store_true",          help="Render once and exit (for Claude one-shot checks)")
    parser.add_argument("--mode",     type=str,   default=None,    choices=["sft", "dpo", "grpo", "vision"],
                        help="Override task type for chart layout (auto-detected from server if omitted)")
    parser.add_argument("--alpha",    type=float, default=0.1,     help="EMA smoothing factor for loss (0 < alpha ≤ 1)")
    args = parser.parse_args()

    if args.once:
        payload = fetch_metrics(args.port)
        if payload:
            draw_dashboard(payload, theme=args.theme, once=True, mode=args.mode, alpha=args.alpha)
        else:
            print(f"No training server on port {args.port} — is training running?")
        return

    print(f"Connecting to Gaslamp training process on http://localhost:{args.port}...")
    try:
        while True:
            payload = fetch_metrics(args.port)
            if payload:
                draw_dashboard(payload, theme=args.theme, mode=args.mode, alpha=args.alpha)
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
