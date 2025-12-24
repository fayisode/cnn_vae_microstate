import os
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import model as _m
import numpy as np
import seeding as _se
import torch
from torch.utils.data import DataLoader

IMAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images"))
FIGURE_SIZE = (8, 5)
DPI = 300


def ensure_directory_exists(filepath: str) -> None:
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)


def save_plot(
    data: Union[List[float], np.ndarray],
    plot_type: str,
    filename: str,
    **kwargs,
) -> None:
    filepath = os.path.join(IMAGE_PATH, filename)
    ensure_directory_exists(filepath)
    fig = plt.figure(figsize=FIGURE_SIZE)
    if plot_type == "loss_curve":
        _plot_loss_curve(data, fig)
    elif plot_type == "loss_histogram":
        _plot_loss_histogram(data, fig)
    elif plot_type == "image_grid":
        _plot_image_grid(data, fig, **kwargs)
    else:
        raise ValueError(f"Invalid plot type: {plot_type}")
    fig.savefig(filepath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_loss_curve(data: Union[List[float], np.ndarray], fig) -> None:
    plt.plot(range(1, len(data) + 1), data, label="Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Gamma Training Loss Curve")
    plt.legend()
    plt.grid(True)


def _plot_loss_histogram(data: Union[List[float], np.ndarray], fig) -> None:
    plt.hist(data, bins=50, alpha=0.75, color="blue")
    plt.title("Loss Distribution Histogram")
    plt.xlabel("Loss Value")
    plt.ylabel("Frequency")
    plt.grid(True)


def _plot_image_grid(
    data: Union[List[float], np.ndarray],
    fig,
    grid_width: int,
    grid_height: int,
) -> None:
    fig.subplots_adjust(hspace=0.15, wspace=0.10)
    for i in range(grid_width * grid_height):
        ax = fig.add_subplot(grid_height, grid_width, i + 1)
        ax.axis("off")
        ax.imshow(data[i, :, :, :])


def save_reconstructed_images(
    original_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    filename: str = "reconstructed_images.png",
    num_samples: int = 10,
) -> None:
    """
    Save reconstructed images in multiple formats:
    1. Traditional grid (original function)
    2. Side-by-side pairs
    3. Individual reconstruction images
    4. Individual original images
    """
    base_filepath = filename.replace(".png", "")
    base_dir = os.path.dirname(base_filepath)

    # Create organized directory structure
    recon_dir = os.path.join(base_dir, "reconstructions")
    grid_dir = os.path.join(recon_dir, "grids")
    pairs_dir = os.path.join(recon_dir, "pairs")
    recon_only_dir = os.path.join(recon_dir, "reconstructed_only")
    orig_only_dir = os.path.join(recon_dir, "original_only")

    for directory in [grid_dir, pairs_dir, recon_only_dir, orig_only_dir]:
        ensure_directory_exists(os.path.join(directory, "dummy.txt"))

    # 1. Traditional grid format
    grid_filepath = os.path.join(grid_dir, "reconstruction_grid.png")
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 2 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    axes[0, 0].set_title("Original", fontsize=14)
    axes[0, 1].set_title("Reconstructed", fontsize=14)

    for i in range(num_samples):
        _plot_image(original_images[i][0], axes[i, 0])
        _plot_image(reconstructed_images[i][0], axes[i, 1])

    plt.tight_layout()
    fig.savefig(grid_filepath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid reconstruction saved to: {grid_filepath}")

    # 2. Side-by-side pairs (each image pair in separate file)
    for i in range(num_samples):
        pair_filepath = os.path.join(pairs_dir, f"pair_{i+1:03d}.png")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        _plot_image(original_images[i][0], ax1)
        ax1.set_title(f"Original {i+1}", fontsize=12)

        _plot_image(reconstructed_images[i][0], ax2)
        ax2.set_title(f"Reconstructed {i+1}", fontsize=12)

        plt.tight_layout()
        fig.savefig(pair_filepath, dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    print(f"Side-by-side pairs saved to: {pairs_dir} ({num_samples} files)")

    # 3. Individual reconstructed images
    for i in range(num_samples):
        recon_filepath = os.path.join(recon_only_dir, f"recon_{i+1:03d}.png")
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        _plot_image(reconstructed_images[i][0], ax)
        ax.set_title(f"Reconstructed {i+1}", fontsize=12)

        plt.tight_layout()
        fig.savefig(recon_filepath, dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    print(
        f"Individual reconstructions saved to: {recon_only_dir} ({num_samples} files)"
    )

    # 4. Individual original images
    for i in range(num_samples):
        orig_filepath = os.path.join(orig_only_dir, f"original_{i+1:03d}.png")
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        _plot_image(original_images[i][0], ax)
        ax.set_title(f"Original {i+1}", fontsize=12)

        plt.tight_layout()
        fig.savefig(orig_filepath, dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    print(f"Individual originals saved to: {orig_only_dir} ({num_samples} files)")

    # Summary
    print(
        f"""
🖼️ Reconstruction images saved in organized structure:
📁 {recon_dir}/
   ├── 📁 grids/          - Traditional side-by-side grid
   ├── 📁 pairs/          - Individual comparison pairs ({num_samples} files)
   ├── 📁 reconstructed_only/ - Only reconstructed images ({num_samples} files) 
   └── 📁 original_only/  - Only original images ({num_samples} files)
"""
    )


def _plot_image(image_tensor: torch.Tensor, ax: plt.Axes) -> None:
    image = image_tensor.cpu().detach().numpy()
    ax.imshow(image, cmap="gray")
    ax.axis("off")


def compute_losses(kld_loss, reconst_loss, loss_function) -> Dict[str, torch.Tensor]:
    return {
        "kld_losses": kld_loss,
        "reconstruct_losses": reconst_loss,
        "epoch_losses": loss_function,
    }


def plot_loss_functions(
    data: Dict[str, List[float]], filename: str = "loss_functions_plot.png"
) -> None:
    filepath = os.path.join(IMAGE_PATH, filename)
    ensure_directory_exists(filepath)

    # Filter out empty data and validate
    plot_data = {}
    for k, v in data.items():
        if (
            isinstance(v, list)
            and len(v) > 0
            and any(x is not None and not np.isnan(x) for x in v)
        ):
            plot_data[k] = v

    if not plot_data:
        print(f"No valid data to plot for {filename}")
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    linestyles = ["-", "--", ":", "-."]
    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    lines_plotted = []
    labels_plotted = []

    for i, (key, value) in enumerate(plot_data.items()):
        try:
            # Clean the data of any NaN/None values
            clean_epochs = []
            clean_values = []
            for j, val in enumerate(value):
                if val is not None and not np.isnan(val):
                    clean_epochs.append(j + 1)
                    clean_values.append(val)

            if len(clean_values) > 0:
                clean_label = key.replace("_", " ").title()
                line = ax.plot(
                    clean_epochs,
                    clean_values,
                    label=clean_label,
                    alpha=0.8,
                    linestyle=linestyles[i % len(linestyles)],
                    color=colors[i % len(colors)],
                    linewidth=2,
                )
                lines_plotted.extend(line)
                labels_plotted.append(clean_label)
        except Exception as e:
            print(f"Warning: Could not plot {key}: {e}")
            continue

    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Values", fontsize=12)
    ax.set_title("Metric History", fontsize=14)

    # Only add legend if we have valid lines plotted
    if lines_plotted and labels_plotted:
        ax.legend(fontsize="medium", loc="best")

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Loss functions plot saved to: {filepath}")


def plot_loss_functions_grid(
    data: Dict[str, List[float]], filename: str = "loss_functions_plot_grid.png"
) -> None:
    filepath = os.path.join(IMAGE_PATH, filename)
    ensure_directory_exists(filepath)

    # Filter out empty lists from data
    plot_data = {k: v for k, v in data.items() if isinstance(v, list) and len(v) > 0}
    num_plots = len(plot_data)

    if num_plots == 0:
        print(f"No data to plot for {filename}")
        return

    ncols = 2
    nrows = (num_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIGURE_SIZE[0] * 1.5, FIGURE_SIZE[1] * nrows / 1.5),
        sharex=True,
    )

    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, (key, value) in enumerate(plot_data.items()):
        ax = axes[i]
        ax.plot(
            range(1, len(value) + 1),
            value,
            label=key.replace("_", " ").title(),
            linewidth=2,
        )
        ax.set_xlabel("Epochs", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.set_title(key.replace("_", " ").title(), fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Metric History", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Loss functions grid plot saved to: {filepath}")


def save_loss_plots(loss_histories: Dict[str, List[float]], name: str) -> None:
    # Save individual loss curves
    for loss_name, loss_values in loss_histories.items():
        if loss_values and len(loss_values) > 0:
            save_plot(loss_values, "loss_curve", filename=f"{loss_name}_{name}.png")

    # Reconstruction and training losses
    reconstruction_losses = {
        "epoch_losses": loss_histories.get("epoch_losses", []),
        "val_losses": loss_histories.get("val_losses", []),
        "reconstruct_losses": loss_histories.get("reconstruct_losses", []),
        "kld_losses": loss_histories.get("kld_losses", []),
        "weighted_kld_loss": loss_histories.get("weighted_kld_loss", []),
        "ssim_scores": loss_histories.get("ssim_scores", []),
    }

    # Updated clustering metrics (removed NMI, ARI, V-measure)
    clustering_metrics = {
        "silhouette_scores": loss_histories.get("silhouette_scores", []),
        "db_scores": loss_histories.get("db_scores", []),
        "ch_scores": loss_histories.get("ch_scores", []),
    }

    # Create plots
    plot_loss_functions(reconstruction_losses, f"reconstruction_{name}.png")
    plot_loss_functions_grid(reconstruction_losses, f"reconstruction_{name}_grid.png")
    plot_loss_functions(clustering_metrics, f"clustering_{name}.png")
    plot_loss_functions_grid(clustering_metrics, f"clustering_{name}_grid.png")


def test_analysis(
    test_loader: DataLoader,
    model: _m.MyModel,
    device: torch.device,
    path,
) -> None:
    t_im, t_la = next(iter(test_loader))
    t_imc = t_im.to(device)
    t_re, t_mu_, t_logvar = model(t_imc)
    save_reconstructed_images(t_im, t_re, path)


def analyze_clusters(
    test_loader: DataLoader,
    model: _m.MyModel,
    clusters: int,
    device: torch.device,
) -> None:
    x_1, y_1 = [], []
    for x, y in iter(test_loader):
        x_1.append(x)
        y_1.append(y)
    x_2 = torch.cat(x_1[:])
    y_2 = torch.cat(y_1[:])
    x_subset = x_2[:1000].to(device)
    gt = y_2[:1000].cpu().numpy()
    if len(gt.shape) > 1:
        gt = gt.flatten()
    with torch.no_grad():
        model_prediction = model.predict(x_subset)
    if isinstance(model_prediction, torch.Tensor):
        model_prediction = model_prediction.cpu().numpy()
    if len(model_prediction.shape) > 1:
        model_prediction = model_prediction.flatten()

    min_length = min(len(gt), len(model_prediction))
    gt = gt[:min_length]
    model_prediction = model_prediction[:min_length]

    for i in range(0, clusters):
        mask = gt == i
        if not np.any(mask):
            print(f"No samples found for ground truth cluster {i}")
            continue
        cluster_preds = model_prediction[mask]
        print(f"Model predictions for ground truth cluster {i}:")
        print(f"Cluster {i} contains {np.sum(mask)} samples")
        unique_preds, counts = np.unique(cluster_preds, return_counts=True)
        print(f"Distribution of predictions for GT cluster {i}:")
        for pred, count in zip(unique_preds, counts):
            print(
                f"  Predicted as cluster {pred}: {count} samples ({count/np.sum(mask)*100:.1f}%)"
            )
        print()


def plot_epoch_results(
    train_loss_arr: List[float],
    val_loss_arr: List[float],
    filename: str = "combined.png",
) -> None:
    ensure_directory_exists(filename)
    fig = plt.figure(figsize=(12, 8))

    if train_loss_arr and len(train_loss_arr) > 0:
        plt.plot(
            range(1, len(train_loss_arr) + 1),
            train_loss_arr,
            label="Train Loss",
            color="green",
            linewidth=2,
        )

    if val_loss_arr and len(val_loss_arr) > 0:
        plt.plot(
            range(1, len(val_loss_arr) + 1),
            val_loss_arr,
            label="Validation Loss",
            color="blue",
            linewidth=2,
        )

    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss Over Epochs", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Epoch results plot saved to: {filename}")


def create_comprehensive_training_plots(
    history: List[Dict], filename: str = "comprehensive_training_analysis.png"
) -> None:
    """Create comprehensive training analysis plots similar to the pandas version"""
    filepath = filename
    ensure_directory_exists(filepath)

    # Convert to easier format
    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss", 0) for h in history]
    val_loss = [h.get("val_loss", 0) for h in history]
    recon_loss = [h.get("recon_loss", 0) for h in history]
    kld_loss = [h.get("kld_loss", 0) for h in history]
    beta = [h.get("beta", 1) for h in history]
    silhouette = [h.get("silhouette_scores", 0) for h in history]
    ssim = [h.get("ssim_score", 0) for h in history]
    ch_scores = [h.get("ch_scores", 0) for h in history]
    db_scores = [h.get("db_scores", 10) for h in history]

    # Calculate KLD * Beta
    kld_beta = [k * b for k, b in zip(kld_loss, beta)]

    # Find best validation loss epoch
    min_val_idx = np.argmin(val_loss)
    min_val_loss_value = val_loss[min_val_idx]
    min_val_epoch = epochs[min_val_idx]

    # Create the plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    # Plot 1: Train and Validation Loss
    ax1.plot(epochs, train_loss, label="Train Loss", color="green", linewidth=2)
    ax1.plot(epochs, val_loss, label="Validation Loss", color="blue", linewidth=2)
    ax1.axvline(
        x=min_val_epoch,
        color="red",
        linestyle="--",
        label=f"Best Val Loss: {min_val_loss_value:.4f} at Epoch {min_val_epoch}",
    )
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Train and Validation Loss", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss Components
    ax2.plot(epochs, train_loss, label="Train Loss", color="green", linewidth=2)
    ax2.plot(epochs, recon_loss, label="Reconstruction Loss", color="blue", linewidth=2)
    ax2.plot(epochs, kld_beta, label="KLD Loss × Beta", color="orange", linewidth=2)
    ax2.axvline(
        x=min_val_epoch,
        color="red",
        linestyle="--",
        label=f"Best Val Loss at Epoch {min_val_epoch}",
    )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Train, Reconstruction, and KLD × Beta Loss", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Normalized Clustering Metrics
    def normalize_metric(values):
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return [0.5] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]

    sil_norm = normalize_metric(silhouette)
    ssim_norm = normalize_metric(ssim)
    ch_norm = normalize_metric(ch_scores)
    db_norm = [1 - v for v in normalize_metric(db_scores)]  # Invert DB scores

    ax3.plot(
        epochs,
        sil_norm,
        label=f"Silhouette (Range: {min(silhouette):.3f} to {max(silhouette):.3f})",
        color="green",
        linewidth=2,
    )
    ax3.plot(
        epochs,
        ssim_norm,
        label=f"SSIM (Range: {min(ssim):.3f} to {max(ssim):.3f})",
        color="blue",
        linewidth=2,
    )
    ax3.plot(
        epochs,
        ch_norm,
        label=f"CH Score (Range: {min(ch_scores):.1f} to {max(ch_scores):.1f})",
        color="orange",
        linewidth=2,
    )
    ax3.plot(
        epochs,
        db_norm,
        label=f"Inverted DB (Range: {min(db_scores):.3f} to {max(db_scores):.3f})",
        color="purple",
        linewidth=2,
    )
    ax3.axvline(
        x=min_val_epoch,
        color="red",
        linestyle="--",
        label=f"Best Val Loss at Epoch {min_val_epoch}",
    )
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_ylabel("Normalized Score (0 to 1)", fontsize=12)
    ax3.set_title("Normalized Clustering Metrics", fontsize=14)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Beta and Learning Dynamics
    ax4.plot(epochs, beta, label="Beta", color="red", linewidth=2)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(
        epochs, kld_loss, label="KLD Loss", color="blue", linestyle="--", linewidth=2
    )
    ax4.axvline(x=min_val_epoch, color="red", linestyle="--", alpha=0.7)
    ax4.set_xlabel("Epoch", fontsize=12)
    ax4.set_ylabel("Beta", color="red", fontsize=12)
    ax4_twin.set_ylabel("KLD Loss", color="blue", fontsize=12)
    ax4.set_title("Beta Scheduling and KLD Loss", fontsize=14)
    ax4.legend(loc="upper left", fontsize=10)
    ax4_twin.legend(loc="upper right", fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Comprehensive training analysis saved to: {filepath}")


def plot_strategy_comparison(
    history: List[Dict], filename: str = "strategy_comparison.png"
) -> None:
    """Plot strategy performance comparison over epochs"""
    filepath = filename
    ensure_directory_exists(filepath)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    epochs = [h["epoch"] for h in history]

    # Strategy scores - Updated for current strategies
    strategy_keys = ["traditional", "clustering_focused"]
    strategy_names = ["Traditional VAE", "Clustering-Focused"]
    strategy_colors = ["blue", "orange"]

    for key, name, color in zip(strategy_keys, strategy_names, strategy_colors):
        try:
            scores = [h["strategy_scores"][key] for h in history]
            ax1.plot(epochs, scores, label=name, linewidth=2, color=color)
        except KeyError:
            print(f"Warning: Strategy '{key}' not found in history")
            continue

    ax1.set_title("Strategy Scores Over Training", fontsize=14)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Strategy Score", fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Clustering metrics vs Validation Loss
    clustering_score = [h.get("composite_clustering_score", 0) for h in history]
    ax2.plot(
        epochs, clustering_score, label="Clustering Score", color="blue", linewidth=2
    )
    ax2_twin = ax2.twinx()
    val_loss = [h.get("val_loss", 0) for h in history]
    ax2_twin.plot(
        epochs,
        val_loss,
        label="Validation Loss",
        color="red",
        linestyle="--",
        linewidth=2,
    )

    ax2.set_title("Clustering Score vs Validation Loss", fontsize=14)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Clustering Score", color="blue", fontsize=12)
    ax2_twin.set_ylabel("Validation Loss", color="red", fontsize=12)
    ax2.legend(loc="upper left", fontsize=10)
    ax2_twin.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Updated Individual clustering metrics (removed NMI, ARI, V-measure)
    metrics = ["silhouette_scores", "db_scores"]
    metric_names = ["Silhouette", "Davies-Bouldin (lower=better)"]
    colors = ["green", "orange", "purple"]

    for metric, name, color in zip(metrics, metric_names, colors):
        try:
            values = [h.get(metric, 0) for h in history]
            if any(v != 0 for v in values):  # Only plot if we have non-zero values
                ax3.plot(epochs, values, label=name, color=color, linewidth=2)
        except KeyError:
            print(f"Warning: Metric '{metric}' not found in history")
            continue

    ax3.set_title("Clustering Metrics Over Training", fontsize=14)
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_ylabel("Metric Value", fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # SSIM and reconstruction quality
    try:
        ssim_scores = [h.get("ssim_score", 0) for h in history]
        if any(s != 0 for s in ssim_scores):
            ax4.plot(epochs, ssim_scores, label="SSIM Score", color="cyan", linewidth=2)

        ax4_twin = ax4.twinx()
        recon_loss = [h.get("recon_loss", 0) for h in history]
        if any(r != 0 for r in recon_loss):
            ax4_twin.plot(
                epochs,
                recon_loss,
                label="Reconstruction Loss",
                color="red",
                linestyle=":",
                linewidth=2,
            )

        ax4.set_title("Reconstruction Quality", fontsize=14)
        ax4.set_xlabel("Epoch", fontsize=12)
        ax4.set_ylabel("SSIM Score", color="cyan", fontsize=12)
        ax4_twin.set_ylabel("Reconstruction Loss", color="red", fontsize=12)
        ax4.legend(loc="upper left", fontsize=10)
        ax4_twin.legend(loc="upper right", fontsize=10)
        ax4.grid(True, alpha=0.3)
    except KeyError:
        ax4.text(
            0.5,
            0.5,
            "SSIM/Reconstruction data not available",
            transform=ax4.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )

    plt.tight_layout()
    plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Strategy comparison plot saved to: {filepath}")


def save_final_comparison_plots(
    results: Dict, filename: str = "final_comparison.png"
) -> None:
    """Create comprehensive final comparison plots"""
    filepath = filename
    ensure_directory_exists(filepath)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    strategies = list(results.keys())
    strategy_names = [results[s]["strategy_name"] for s in strategies]

    # Test loss comparison
    test_losses = [results[s]["test_loss"] for s in strategies]
    bars1 = ax1.bar(strategy_names, test_losses, alpha=0.7, color=["blue", "orange"])
    ax1.set_title("Test Loss by Strategy", fontsize=14)
    ax1.set_ylabel("Test Loss", fontsize=12)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # Add value labels on bars
    for bar, value in zip(bars1, test_losses):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(test_losses) * 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Clustering scores
    clustering_scores = [results[s]["clustering_score"] for s in strategies]
    bars2 = ax2.bar(
        strategy_names, clustering_scores, alpha=0.7, color=["green", "red"]
    )
    ax2.set_title("Clustering Score by Strategy", fontsize=14)
    ax2.set_ylabel("Clustering Score", fontsize=12)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    # Add value labels on bars
    for bar, value in zip(bars2, clustering_scores):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(clustering_scores) * 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Multiple metrics comparison
    metrics = ["silhouette_scores", "ssim_score", "db_scores"]
    metric_names = ["Silhouette", "SSIM", "Davies-Bouldin"]

    x = np.arange(len(metric_names))
    width = 0.35

    # Get metric values for each strategy
    strategy1_values = [results[strategies[0]].get(m, 0) for m in metrics]
    strategy2_values = [
        results[strategies[1]].get(m, 0) if len(strategies) > 1 else 0 for m in metrics
    ]

    # Normalize Davies-Bouldin (lower is better, so invert)
    if len(strategy1_values) >= 4:
        strategy1_values[3] = 1 / (1 + strategy1_values[3])  # Invert DB score
    if len(strategy2_values) >= 4:
        strategy2_values[3] = 1 / (1 + strategy2_values[3])  # Invert DB score

    bars3_1 = ax3.bar(
        x - width / 2,
        strategy1_values,
        width,
        label=strategy_names[0],
        alpha=0.7,
        color="blue",
    )
    if len(strategies) > 1:
        bars3_2 = ax3.bar(
            x + width / 2,
            strategy2_values,
            width,
            label=strategy_names[1],
            alpha=0.7,
            color="orange",
        )

    ax3.set_title("Metric Comparison Across Strategies", fontsize=14)
    ax3.set_xlabel("Metrics", fontsize=12)
    ax3.set_ylabel("Score", fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metric_names)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for bars in [bars3_1] + ([bars3_2] if len(strategies) > 1 else []):
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Best epochs and training summary
    best_epochs = [results[s]["best_epoch"] for s in strategies]
    bars4 = ax4.bar(strategy_names, best_epochs, alpha=0.7, color=["purple", "brown"])
    ax4.set_title("Best Epoch by Strategy", fontsize=14)
    ax4.set_ylabel("Best Epoch", fontsize=12)
    plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")

    # Add value labels on bars
    for bar, value in zip(bars4, best_epochs):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(best_epochs) * 0.01,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Final comparison plot saved to: {filepath}")


def create_detailed_metrics_report(
    detailed_results: Dict, filename: str = "detailed_metrics_report.png"
) -> None:
    """Create a detailed visual report from the detailed JSON results"""
    filepath = filename
    ensure_directory_exists(filepath)

    fig = plt.figure(figsize=(24, 16))

    # Create a 3x3 grid of subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    strategies = list(detailed_results.keys())
    strategy_names = [
        detailed_results[s]["strategy_info"]["strategy_name"] for s in strategies
    ]
    colors = ["blue", "orange", "green", "red"][: len(strategies)]

    # 1. Test vs Validation Loss Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    test_losses = [
        detailed_results[s]["test_performance"]["test_loss"] for s in strategies
    ]
    val_losses = [
        detailed_results[s]["validation_performance"]["val_loss"] for s in strategies
    ]

    x = np.arange(len(strategy_names))
    width = 0.35
    ax1.bar(
        x - width / 2, test_losses, width, label="Test Loss", alpha=0.7, color="red"
    )
    ax1.bar(
        x + width / 2,
        val_losses,
        width,
        label="Validation Loss",
        alpha=0.7,
        color="blue",
    )
    ax1.set_title("Test vs Validation Loss", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategy_names, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Reconstruction Quality (RECON + KLD + SSIM)
    ax2 = fig.add_subplot(gs[0, 1])
    recon_losses = [
        detailed_results[s]["test_performance"]["recon_loss"] for s in strategies
    ]
    kld_losses = [
        detailed_results[s]["test_performance"]["kld_loss"] for s in strategies
    ]
    ssim_scores = [
        detailed_results[s]["test_performance"]["ssim_score"] for s in strategies
    ]

    x = np.arange(len(strategy_names))
    width = 0.25
    ax2.bar(
        x - width, recon_losses, width, label="Recon Loss", alpha=0.7, color="purple"
    )
    ax2.bar(x, kld_losses, width, label="KLD Loss", alpha=0.7, color="orange")
    ax2_twin = ax2.twinx()
    ax2_twin.bar(x + width, ssim_scores, width, label="SSIM", alpha=0.7, color="green")

    ax2.set_title("Reconstruction Quality", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategy_names, rotation=45, ha="right")
    ax2.set_ylabel("Loss Values", color="purple")
    ax2_twin.set_ylabel("SSIM Score", color="green")
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # 3. Clustering Performance
    ax3 = fig.add_subplot(gs[0, 2])
    silhouette = [
        detailed_results[s]["test_performance"]["silhouette_scores"] for s in strategies
    ]

    x = np.arange(len(strategy_names))
    width = 0.35
    ax3.bar(
        x - width / 2, silhouette, width, label="Silhouette", alpha=0.7, color="cyan"
    )
    ax3.set_title("Clustering Performance", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategy_names, rotation=45, ha="right")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Strategy Weights Visualization
    ax4 = fig.add_subplot(gs[1, 0])
    recon_weights = [
        detailed_results[s]["strategy_info"]["weight_recon"] for s in strategies
    ]
    cluster_weights = [
        detailed_results[s]["strategy_info"]["weight_clustering"] for s in strategies
    ]

    x = np.arange(len(strategy_names))
    width = 0.35
    ax4.bar(
        x - width / 2,
        recon_weights,
        width,
        label="Recon Weight",
        alpha=0.7,
        color="blue",
    )
    ax4.bar(
        x + width / 2,
        cluster_weights,
        width,
        label="Clustering Weight",
        alpha=0.7,
        color="red",
    )
    ax4.set_title("Strategy Weight Configuration", fontsize=12, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(strategy_names, rotation=45, ha="right")
    ax4.set_ylabel("Weight Value")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Best Epochs and Training Efficiency
    ax5 = fig.add_subplot(gs[1, 1])
    best_epochs = [
        detailed_results[s]["strategy_info"]["best_epoch"] for s in strategies
    ]
    strategy_scores = [
        detailed_results[s]["training_history"]["best_strategy_score"]
        for s in strategies
    ]

    ax5.bar(strategy_names, best_epochs, alpha=0.7, color=colors)
    ax5.set_title("Training Efficiency", fontsize=12, fontweight="bold")
    ax5.set_ylabel("Best Epoch")
    ax5.set_xlabel("Strategy")
    plt.setp(ax5.get_xticklabels(), rotation=45, ha="right")

    # Add values on bars
    for i, (epoch, score) in enumerate(zip(best_epochs, strategy_scores)):
        ax5.text(
            i,
            epoch + max(best_epochs) * 0.01,
            f"Epoch {epoch}\nScore: {score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax5.grid(True, alpha=0.3)

    # 6. Normalized Metrics Radar-like Comparison
    ax6 = fig.add_subplot(gs[1, 2])
    for i, strategy in enumerate(strategies):
        norm_metrics = detailed_results[strategy]["normalized_metrics"]
        metrics_names = list(norm_metrics.keys())
        metrics_values = list(norm_metrics.values())

        x_pos = np.arange(len(metrics_names))
        ax6.plot(
            x_pos,
            metrics_values,
            "o-",
            label=strategy_names[i],
            color=colors[i],
            linewidth=2,
            markersize=6,
        )

    ax6.set_title("Normalized Metrics Comparison", fontsize=12, fontweight="bold")
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(
        [name.replace("_norm", "").title() for name in metrics_names],
        rotation=45,
        ha="right",
    )
    ax6.set_ylabel("Normalized Score (0-1)")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 1)

    # 7. Davies-Bouldin and Calinski-Harabasz Scores
    ax7 = fig.add_subplot(gs[2, 0])
    db_scores = [
        detailed_results[s]["test_performance"]["db_scores"] for s in strategies
    ]
    ch_scores = [
        detailed_results[s]["test_performance"]["ch_scores"] for s in strategies
    ]

    ax7.bar(
        x - width / 2,
        db_scores,
        width,
        label="Davies-Bouldin (lower=better)",
        alpha=0.7,
        color="red",
    )
    ax7_twin = ax7.twinx()
    ax7_twin.bar(
        x + width / 2,
        ch_scores,
        width,
        label="Calinski-Harabasz (higher=better)",
        alpha=0.7,
        color="green",
    )

    ax7.set_title("Advanced Clustering Metrics", fontsize=12, fontweight="bold")
    ax7.set_xticks(x)
    ax7.set_xticklabels(strategy_names, rotation=45, ha="right")
    ax7.set_ylabel("Davies-Bouldin Score", color="red")
    ax7_twin.set_ylabel("Calinski-Harabasz Score", color="green")
    ax7.legend(loc="upper left")
    ax7_twin.legend(loc="upper right")
    ax7.grid(True, alpha=0.3)

    # 8. Beta Values Comparison
    ax8 = fig.add_subplot(gs[2, 1])
    test_betas = [detailed_results[s]["test_performance"]["beta"] for s in strategies]
    val_betas = [
        detailed_results[s]["validation_performance"]["beta"] for s in strategies
    ]

    x = np.arange(len(strategy_names))
    width = 0.35
    ax8.bar(
        x - width / 2, test_betas, width, label="Test Beta", alpha=0.7, color="purple"
    )
    ax8.bar(
        x + width / 2,
        val_betas,
        width,
        label="Validation Beta",
        alpha=0.7,
        color="pink",
    )
    ax8.set_title("Beta Values Comparison", fontsize=12, fontweight="bold")
    ax8.set_xticks(x)
    ax8.set_xticklabels(strategy_names, rotation=45, ha="right")
    ax8.set_ylabel("Beta Value")
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Summary Text Box
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")

    # Create summary text
    summary_text = "TRAINING SUMMARY\n" + "=" * 20 + "\n\n"

    for i, strategy in enumerate(strategies):
        info = detailed_results[strategy]
        summary_text += f"{info['strategy_info']['strategy_name']}:\n"
        summary_text += f"  • Best Epoch: {info['strategy_info']['best_epoch']}\n"
        summary_text += f"  • Test Loss: {info['test_performance']['test_loss']:.4f}\n"
        summary_text += f"  • Clustering Score: {info['test_performance']['clustering_score']:.4f}\n"
        summary_text += (
            f"  • Silhouette: {info['test_performance']['silhouette_scores']:.4f}\n"
        )
        summary_text += f"  • SSIM: {info['test_performance']['ssim_score']:.4f}\n"

        if info["test_performance"].get("clustering_failed", False):
            summary_text += "  ⚠️ Clustering Failed\n"

        summary_text += "\n"

    # Add model configuration
    config = detailed_results[strategies[0]]["model_config"]
    summary_text += f"MODEL CONFIG:\n"
    summary_text += f"  • Learning Rate: {config['learning_rate']}\n"
    summary_text += f"  • Total Epochs: {config['total_epochs']}\n"
    summary_text += f"  • Patience: {config['patience']}\n"
    summary_text += f"  • Pretrain Epochs: {config['pretrain_epochs']}\n"

    ax9.text(
        0.05,
        0.95,
        summary_text,
        transform=ax9.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )

    plt.suptitle(
        "Detailed Strategy Analysis Report", fontsize=20, fontweight="bold", y=0.98
    )
    plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Detailed metrics report saved to: {filepath}")


def create_publication_ready_plots(
    detailed_results: Dict,
    epoch_history: List[Dict],
    filename_prefix: str = "publication_ready",
) -> None:
    """Create publication-ready plots with high quality formatting"""

    # Set publication style
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "serif",
            "axes.linewidth": 1.2,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
        }
    )

    # 1. Training Curves (Publication Quality)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    epochs = [h["epoch"] for h in epoch_history]
    train_loss = [h.get("train_loss", 0) for h in epoch_history]
    val_loss = [h.get("val_loss", 0) for h in epoch_history]

    ax1.plot(epochs, train_loss, label="Training Loss", linewidth=2.5, color="#1f77b4")
    ax1.plot(epochs, val_loss, label="Validation Loss", linewidth=2.5, color="#ff7f0e")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # Find and mark best validation loss
    min_val_idx = np.argmin(val_loss)
    ax1.axvline(
        x=epochs[min_val_idx],
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Best Val Loss (Epoch {epochs[min_val_idx]})",
    )

    # Clustering metrics evolution
    silhouette = [h.get("silhouette_scores", 0) for h in epoch_history]
    ssim = [h.get("ssim_score", 0) for h in epoch_history]

    ax2.plot(
        epochs, silhouette, label="Silhouette Score", linewidth=2.5, color="#2ca02c"
    )
    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        epochs, ssim, label="SSIM Score", linewidth=2.5, color="#d62728", linestyle="--"
    )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Silhouette Score", color="#2ca02c")
    ax2_twin.set_ylabel("SSIM Score", color="#d62728")
    ax2.set_title("Clustering Quality Evolution")
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Strategy Comparison (Publication Quality)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    strategies = list(detailed_results.keys())
    strategy_names = [
        detailed_results[s]["strategy_info"]["strategy_name"] for s in strategies
    ]

    # Test performance comparison
    test_losses = [
        detailed_results[s]["test_performance"]["test_loss"] for s in strategies
    ]
    clustering_scores = [
        detailed_results[s]["test_performance"]["clustering_score"] for s in strategies
    ]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][: len(strategies)]

    bars1 = ax1.bar(
        strategy_names, test_losses, color=colors, alpha=0.8, edgecolor="black"
    )
    ax1.set_title("Test Loss Comparison")
    ax1.set_ylabel("Test Loss")
    ax1.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, value in zip(bars1, test_losses):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(test_losses) * 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    bars2 = ax2.bar(
        strategy_names, clustering_scores, color=colors, alpha=0.8, edgecolor="black"
    )
    ax2.set_title("Clustering Score Comparison")
    ax2.set_ylabel("Clustering Score")
    ax2.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, value in zip(bars2, clustering_scores):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(clustering_scores) * 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Detailed metrics comparison
    metrics = ["silhouette_scores", "ssim_score"]
    metric_labels = ["Silhouette Score", "SSIM Score"]

    x = np.arange(len(metric_labels))
    width = 0.35

    if len(strategies) >= 2:
        strategy1_values = [
            detailed_results[strategies[0]]["test_performance"][m] for m in metrics
        ]
        strategy2_values = [
            detailed_results[strategies[1]]["test_performance"][m] for m in metrics
        ]

        bars3_1 = ax3.bar(
            x - width / 2,
            strategy1_values,
            width,
            label=strategy_names[0],
            color=colors[0],
            alpha=0.8,
            edgecolor="black",
        )
        bars3_2 = ax3.bar(
            x + width / 2,
            strategy2_values,
            width,
            label=strategy_names[1],
            color=colors[1],
            alpha=0.8,
            edgecolor="black",
        )

        ax3.set_title("Detailed Metrics Comparison")
        ax3.set_xlabel("Metrics")
        ax3.set_ylabel("Score")
        ax3.set_xticks(x)
        ax3.set_xticklabels(metric_labels)
        ax3.legend()

        # Add value labels
        for bars in [bars3_1, bars3_2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.005,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    # Training efficiency
    best_epochs = [
        detailed_results[s]["strategy_info"]["best_epoch"] for s in strategies
    ]
    bars4 = ax4.bar(
        strategy_names, best_epochs, color=colors, alpha=0.8, edgecolor="black"
    )
    ax4.set_title("Training Efficiency")
    ax4.set_ylabel("Best Epoch")
    ax4.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, value in zip(bars4, best_epochs):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(best_epochs) * 0.01,
            f"{value}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        f"{filename_prefix}_strategy_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"Publication-ready plots saved with prefix: {filename_prefix}")

    # Reset matplotlib settings
    plt.rcdefaults()
