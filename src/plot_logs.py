import os
import json
import argparse
import numpy as np

def plot_training_metrics(history_file: str = "logs/training_history.json", output_dir: str = "logs/plots"):
    """
    Reads structured training history JSON and plots publication-ready loss and metric curves.
    """
    gdrive_history = "/content/drive/MyDrive/Tenuto/logs/training_history.json"
    target_file = history_file
    
    if not os.path.exists(target_file) and os.path.exists(gdrive_history):
        target_file = gdrive_history

    if not os.path.exists(target_file):
        print(f"[TenutoPlot] ❌ Log file '{history_file}' not found. Cannot plot metrics.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg") # Non-interactive headless backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[TenutoPlot] ❌ matplotlib is required for plotting. Install via `pip install matplotlib`.")
        return

    with open(target_file, "r") as f:
        history = json.load(f)

    if not history:
        print("[TenutoPlot] Log history is empty.")
        return

    os.makedirs(output_dir, exist_ok=True)

    epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]
    train_loss = [h.get("train_loss", 0.0) for h in history]
    val_loss = [h.get("val_loss", 0.0) for h in history]
    
    train_vel = [h.get("train_vel_rmse", 0.0) for h in history]
    val_vel = [h.get("val_vel_rmse", 0.0) for h in history]

    train_timing = [h.get("train_timing_ms", 0.0) for h in history]
    val_timing = [h.get("val_timing_ms", 0.0) for h in history]

    lr_history = [h.get("learning_rate", 0.0) for h in history]

    plt.style.use("ggplot")

    # 1. Total Loss Plot (Train vs Val)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2.5, marker="o")
    plt.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", linewidth=2.5, marker="s")
    plt.title("Tenuto Transformer: Total Training & Validation Loss", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Total Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()

    # 2. Velocity RMSE Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_vel, label="Train Vel RMSE", color="#2ca02c", linewidth=2.5, marker="o")
    plt.plot(epochs, val_vel, label="Val Vel RMSE", color="#d62728", linewidth=2.5, marker="s")
    plt.title("Velocity Accuracy (RMSE in MIDI Steps)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Velocity RMSE (MIDI Steps)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    vel_plot_path = os.path.join(output_dir, "velocity_rmse.png")
    plt.savefig(vel_plot_path, dpi=300)
    plt.close()

    # 3. Micro-Timing Huber Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_timing, label="Train Timing Huber", color="#9467bd", linewidth=2.5, marker="o")
    plt.plot(epochs, val_timing, label="Val Timing Huber", color="#8c564b", linewidth=2.5, marker="s")
    plt.title("Micro-Timing Offset Accuracy (ms)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Timing Huber Error (ms)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    timing_plot_path = os.path.join(output_dir, "micro_timing_ms.png")
    plt.savefig(timing_plot_path, dpi=300)
    plt.close()

    # 4. Learning Rate Schedule Plot
    if any(lr > 0 for lr in lr_history):
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, lr_history, label="Learning Rate", color="#17becf", linewidth=2.0)
        plt.title("Optax Learning Rate Schedule (Warmup + Cosine Decay)", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Learning Rate", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        lr_plot_path = os.path.join(output_dir, "learning_rate_schedule.png")
        plt.savefig(lr_plot_path, dpi=300)
        plt.close()

    print(f"[TenutoPlot] ✅ Publication-ready metric plots successfully saved to '{output_dir}'!")
    print(f"  • Loss Curves    : {loss_plot_path}")
    print(f"  • Velocity RMSE  : {vel_plot_path}")
    print(f"  • Micro Timing ms: {timing_plot_path}")

    # Sync plots to Google Drive if available
    gdrive_plots = "/content/drive/MyDrive/Tenuto/logs/plots"
    if os.path.exists("/content/drive/MyDrive"):
        os.makedirs(gdrive_plots, exist_ok=True)
        import shutil
        for f_name in ["loss_curves.png", "velocity_rmse.png", "micro_timing_ms.png", "learning_rate_schedule.png"]:
            local_f = os.path.join(output_dir, f_name)
            if os.path.exists(local_f):
                shutil.copy(local_f, os.path.join(gdrive_plots, f_name))
        print(f"[TenutoPlot] Synced metric plots to Google Drive: {gdrive_plots}")

def main(args=None):
    parser = argparse.ArgumentParser(description="Plot Tenuto Training Metrics")
    parser.add_argument("--history_file", type=str, default="logs/training_history.json", help="Path to training history JSON")
    parser.add_argument("--output_dir", type=str, default="logs/plots", help="Directory to save plot images")
    parsed_args = parser.parse_args(args)
    plot_training_metrics(parsed_args.history_file, parsed_args.output_dir)

if __name__ == "__main__":
    main()
