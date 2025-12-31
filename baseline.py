import numpy as np
import mne
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import json
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from pycrostates.cluster import ModKMeans
from pycrostates.io import ChData


class BaselineHandler:
    def __init__(self, n_clusters, device, logger, output_dir):
        self.n_clusters = n_clusters
        self.device = device
        self.logger = logger
        self.output_dir = Path(output_dir) / "baseline_modkmeans"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ModKMeans
        # n_init=10 restarts clustering 10 times to find the global optimum
        self.modk = ModKMeans(
            n_clusters=n_clusters, n_init=10, max_iter=100, random_state=42
        )
        self.fitted = False
        self.cluster_centers_ = None

    def _extract_data_from_loader(self, loader):
        """
        Extracts data from PyTorch DataLoader and flattens it.

        CRITICAL NOTE:
        Standard microstate analysis typically uses:
        1. 32-64 raw electrode channels.
        2. Only GFP Peaks (high signal-to-noise ratio).

        However, to strictly compare this Baseline against your VAE,
        we must use the EXACT SAME INPUTS. Since your VAE takes
        40x40 interpolated images, we flatten these to (1600,) vectors.

        This treats every pixel as a 'channel' for the K-Means algorithm.
        """
        self.logger.info("Extracting VAE input data for Baseline K-Means...")
        data_list = []

        # We cannot easily filter for GFP peaks here because the DataLoader
        # is likely shuffled (temporal structure is lost).
        # We must cluster *all* data points, just like the VAE does.
        for data, _ in tqdm(loader, desc="Loading Baseline Data"):
            # Input shape: (Batch, 1, 40, 40) -> Flatten to (Batch, 1600)
            d_flat = data.view(data.size(0), -1).numpy()
            data_list.append(d_flat)

        X = np.concatenate(data_list, axis=0)
        return X

    def fit(self, train_loader):
        """
        Fits Modified K-Means on the training data.
        """
        self.logger.info("=" * 60)
        self.logger.info(f"STARTING BASELINE: Modified K-Means (K={self.n_clusters})")
        self.logger.info("=" * 60)

        #  Get Data (N_samples, N_features)
        X = self._extract_data_from_loader(train_loader)
        n_samples, n_features = X.shape

        # Convert to pycrostates ChData format
        # pycrostates expects (n_channels, n_samples).
        # Here, our 'channels' are the 1600 pixels of the topomap.
        self.logger.info(f"Creating ChData from {n_features} features (pixels)...")

        # We create dummy channel names "0", "1", ... "1599"
        info = mne.create_info(
            ch_names=[str(i) for i in range(n_features)], sfreq=100, ch_types="eeg"
        )

        # Transpose X to match (Channels, Samples)
        ch_data = ChData(X.T, info=info)

        #  Fit
        self.logger.info(f"Fitting ModKMeans on {n_samples} samples...")
        try:
            # We fit on the ChData object.
            # This is functionally identical to fitting on a 'Peak Process' output,
            # except our vectors are dense images, not sparse electrodes.
            self.modk.fit(ch_data, n_jobs=-1)

            # Store centers: Shape (n_clusters, n_features)
            self.cluster_centers_ = self.modk.cluster_centers_
            self.fitted = True
            self.logger.info("✅ Baseline fitting complete.")

        except Exception as e:
            self.logger.error(f"Baseline fitting failed: {e}")

    def evaluate(self, test_loader):
        """
        Evaluates the baseline using the same metrics as the VAE.
        """
        if not self.fitted:
            self.logger.warning("Baseline not fitted. Skipping evaluation.")
            return {}

        self.logger.info("Evaluating Baseline Performance...")
        X = self._extract_data_from_loader(test_loader)

        # --- Prediction Logic (ModKMeans style) ---
        #  Normalize Input and Centers (Cosine distance relies on direction, not amplitude)
        # Add epsilon to avoid division by zero
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        C_norm = self.cluster_centers_ / (
            np.linalg.norm(self.cluster_centers_, axis=1, keepdims=True) + 1e-8
        )

        #  Calculate Correlation (Dot product of normalized vectors)
        correlation = np.dot(X_norm, C_norm.T)

        #  Ignore Polarity (Absolute correlation)
        dist = np.abs(correlation)

        #  Assign Labels (Highest correlation)
        labels = np.argmax(dist, axis=1)

        # --- Calculate Metrics ---

        #  GEV (Global Explained Variance)
        # GFP = Standard deviation across space (features)
        gfp = np.std(X, axis=1)
        gfp_sum_sq = np.sum(gfp**2)

        # Correlation of the specific assigned map
        assigned_corr = dist[np.arange(len(X)), labels]

        # GEV Formula
        gev = np.sum((gfp * assigned_corr) ** 2) / (gfp_sum_sq + 1e-8)

        # B. Clustering Metrics (sklearn)
        # We use X_norm to be consistent with how ModKMeans views the data (ignoring amplitude)
        try:
            # Subsample for Silhouette to prevent OOM on large datasets
            sample_idx = np.random.choice(
                len(X_norm), size=min(5000, len(X_norm)), replace=False
            )
            sil = silhouette_score(X_norm[sample_idx], labels[sample_idx])

            db = davies_bouldin_score(X_norm, labels)
            ch = calinski_harabasz_score(X_norm, labels)
        except Exception as e:
            self.logger.warning(f"Could not calc sklearn metrics: {e}")
            sil, db, ch = -1, 10, 0

        metrics = {
            "strategy_name": "Baseline (ModKMeans)",
            "gev": float(gev),
            "silhouette_scores": float(sil),
            "db_scores": float(db),
            "ch_scores": float(ch),
            "n_clusters": self.n_clusters,
        }

        self.logger.info(f"📊 Baseline Results:")
        self.logger.info(f"   GEV: {gev:.4f}")
        self.logger.info(f"   Silhouette: {sil:.4f}")
        self.logger.info(f"   DB Index: {db:.4f}")

        with open(self.output_dir / "baseline_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        return metrics

    def visualize_centroids(self, input_shape=(40, 40)):
        """
        Reshapes the flat 1600-dim centroids back to 40x40 images and saves them.
        """
        if not self.fitted:
            return

        self.logger.info("Visualizing Baseline Centroids...")
        n_cols = 4
        n_rows = int(np.ceil(self.n_clusters / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        centers = self.cluster_centers_

        for i in range(self.n_clusters):
            # Reshape flat vector back to image for visualization
            img = centers[i].reshape(input_shape)

            # Normalize for visualization
            vmax = np.percentile(np.abs(img), 99)

            axes[i].imshow(img, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
            axes[i].set_title(f"Baseline Map {i+1}")
            axes[i].axis("off")

        for i in range(self.n_clusters, len(axes)):
            axes[i].axis("off")

        plt.suptitle(f"Modified K-Means Centroids (K={self.n_clusters})", fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / "baseline_centroids.png", dpi=300)
        plt.close()
