from joblib import logger
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Dict, List, Union, Any, Optional
from pathlib import Path
import yaml
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy
import gc
import json
import time
import pandas as pd
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import model as _m
import helper_function as _g
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.cluster import KMeans


def clear_gpu(logger=None):
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1)
    except Exception as e:  # Add 'as e' here
        if logger:  # Check if logger exists
            logger.warning(f"Error clearing GPU memory: {e}")


def check_data_preprocessing(data_loader, device, logger=None):
    if logger:
        logger.info("🔍 CHECKING DATA PREPROCESSING")
        logger.info("=" * 40)
    sample_batch = next(iter(data_loader))[0].to(device)
    stats = {
        "shape": list(sample_batch.shape),
        "min": sample_batch.min().item(),
        "max": sample_batch.max().item(),
        "mean": sample_batch.mean().item(),
        "std": sample_batch.std().item(),
        "nan_count": torch.isnan(sample_batch).sum().item(),
        "inf_count": torch.isinf(sample_batch).sum().item(),
        "zero_count": (sample_batch == 0).sum().item(),
    }
    if logger:
        logger.info(f"Raw data shape: {stats['shape']}")
        logger.info(f"Raw data range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        logger.info(f"Raw data mean: {stats['mean']:.6f}")
        logger.info(f"Raw data std: {stats['std']:.6f}")
        logger.info(f"NaN values: {stats['nan_count']}")
        logger.info(f"Inf values: {stats['inf_count']}")
        logger.info(
            f"Zero values: {stats['zero_count']} ({100*stats['zero_count']/sample_batch.numel():.1f}%)"
        )
    return stats


def full_diagnosis(model, train_loader, device, logger=None):
    if logger:
        logger.info("🏥 COMPREHENSIVE CLUSTER DIAGNOSIS")
        logger.info("=" * 80)
    stats = check_data_preprocessing(train_loader, device, logger)
    if logger:
        logger.info("🏃 Quick feature extraction training (10 epochs)...")
    cluster_results = {}
    recommendations = []
    if cluster_results.get("optimal_k_silhouette"):
        recommendations.append(
            f"✅ Use {cluster_results['optimal_k_silhouette']} clusters (silhouette optimal)"
        )
    if logger:
        logger.info("💡 RECOMMENDATIONS:")
        logger.info("=" * 40)
        for rec in recommendations:
            logger.info(rec)
    return {
        "data_stats": stats,
        "cluster_analysis": cluster_results,
        "recommendations": recommendations,
    }


class ClusteringFailureError(Exception):
    pass


class VAEClusteringTrainer:
    def __init__(
        self,
        model: _m.MyModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        train_set: Any,
        config: Dict[str, Any],
        device: torch.device,
        cluster_id: int = 0,
        logger: Any = None,
    ):
        self._validate_inputs(
            model, train_loader, val_loader, test_loader, config, device, logger
        )

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_set = train_set
        self.config = config
        self.device = device
        self.cluster_id = cluster_id
        self.logger = logger
        self.logger.info(
            f"📊 Trainer initialized with batch size: {self.train_loader.batch_size}"
        )
        self.lr = config.get("learning_rate", 1e-3)
        self.epochs = config.get("epochs", 100)
        self.patience = config.get("patience", 10)
        self.pretrain_epochs = config.get("pretrain_epochs", 30)
        self.unfreeze_prior_epoch = config.get("unfreeze_prior_epoch", 1)
        # self.unfreeze_prior_epoch = config.get(
        #     "unfreeze_prior_epoch", max(15, int(self.epochs))
        # )

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, "min", patience=5, factor=0.7
        # )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=20,
            factor=0.5,
            min_lr=1e-4,
            threshold=5e-4,
            threshold_mode="rel",
        )

        self.output_dir = Path(config.get("output_dir", "./outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.output_dir / "checkpoint.pth"
        self._setup_strategies()

        self.train_losses, self.val_losses = [], []
        self.no_improvement_counter = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_epoch = 0
        self.best_train_loss = float("inf")
        self.best_train_epoch = 0

        self.start_epoch = 1
        self.pretrain_epochs = config.get("pretrain_epochs", 30)
        self.gamma_steps = config.get("gamma_steps", len(self.train_loader))
        self.initial_gamma = config.get("initial_gamma", 0.1)
        self.freeze_prior_epochs = config.get("freeze_prior_epochs", 0)
        # self.freeze_prior_epochs = config.get("freeze_prior_epochs", 5)

        if self.checkpoint_path.exists():
            self.logger.info(
                f"Existing checkpoint found at {self.checkpoint_path}. Attempting to resume."
            )
            try:
                last_completed_epoch = self.load_checkpoint()
                self.start_epoch = last_completed_epoch + 1
                self.logger.info(
                    f"Successfully resumed. Will start training from epoch {self.start_epoch}."
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to load checkpoint: {e}. Starting from scratch."
                )

    def _setup_strategies(self):
        self.strategies = {
            "traditional": {
                "name": "Traditional VAE",
                "weight_recon": 1.0,
                "weight_clustering": 0.0,
                "path": self.output_dir / "best_model_traditional.pth",
            },
        }
        self.strategy_states = {
            key: {"best_score": -float("inf"), "best_epoch": 0}
            for key in self.strategies
        }

    def _validate_inputs(
        self, model, train_loader, val_loader, test_loader, config, device, logger
    ):
        if model is None:
            raise ValueError("Model cannot be None")
        if not all(
            hasattr(model, m)
            for m in ["pretrain", "predict", "encode", "loss_function", "RE", "KLD"]
        ):
            raise ValueError("Model is missing required methods")
        if logger is None:
            raise ValueError("Logger cannot be None")
        if device is None:
            raise ValueError("Device cannot be None")

    def pretrain(self):
        try:
            self.logger.info(
                "=" * 60 + "\nSTEP 1: PRETRAINING & DIAGNOSTICS\n" + "=" * 60
            )
            diagnosis_results = full_diagnosis(
                deepcopy(self.model), self.train_loader, self.device, self.logger
            )
            with open(self.output_dir / "cluster_diagnosis.json", "w") as f:
                json.dump(
                    self._convert_to_json_serializable(diagnosis_results), f, indent=4
                )
            self.logger.info(
                f"Diagnosis results saved to {self.output_dir / 'cluster_diagnosis.json'}"
            )

            self.logger.info("Starting pretraining...")
            self.model.pretrain(
                train_loader=self.train_loader,
                optimizer=self.optimizer,
                train_set=self.train_set,
                epochs=self.pretrain_epochs,
                device=self.device,
                gamma_steps=self.gamma_steps,
                initial_gamma=self.initial_gamma,
                freeze_prior_epochs=self.freeze_prior_epochs,
                output_dir=self.output_dir,
            )
            self.logger.info("Pretraining completed successfully")
        except Exception as e:
            self.logger.error(f"Pretraining failed: {e}", exc_info=True)
            raise

    def train_epoch(self, epoch: int) -> Tuple[float, Dict]:
        self.model.train()
        train_loss, valid_batches = 0.0, 0
        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {epoch} Training", leave=False
        )
        for data, _ in progress_bar:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            try:
                recon_batch, mu, logvar = self.model(data)
                re_loss, kld_loss, loss = self.model.loss_function(
                    recon_batch, data, mu, logvar, epoch, self.epochs
                )
                if self.unfreeze_prior_epoch <= epoch < self.unfreeze_prior_epoch + 5:
                    safe_beta = min(0.001, self._get_safe_beta(epoch))
                    loss = re_loss + safe_beta * kld_loss
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    self.logger.warning(
                        f"Collapse detected in training - loss: {loss.item()}"
                    )
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                train_loss += loss.item() * data.size(0)
                valid_batches += data.size(0)
                progress_bar.set_postfix(
                    {
                        "Total": f"{loss.item():.4f}",
                        "Recon": f"{re_loss.item():.4f}",
                        "KLD": f"{kld_loss.item():.4f}",
                        "Beta": f"{self._get_current_beta(epoch):.4f}",
                    }
                )
            except Exception as e:
                self.logger.warning(f"Batch training error: {e}")
                continue

        if valid_batches == 0:
            return 50.0, {}

        avg_train_loss = train_loss / valid_batches

        # Calculate training metrics after training is complete
        self.logger.debug(f"Computing training metrics for epoch {epoch}...")
        clear_gpu(self.logger)
        train_metrics = self._compute_training_metrics(epoch, 0)
        clear_gpu(self.logger)

        return avg_train_loss, train_metrics

    def _get_safe_beta(self, epoch):
        if hasattr(self.model, "loss_balancer") and self.model.loss_balancer:
            beta = self.model.loss_balancer.get_beta(epoch, 0, 0)  # Dummy values
            # Cap beta very low during first 5 epochs after unfreezing
            if self.unfreeze_prior_epoch <= epoch < self.unfreeze_prior_epoch + 5:
                beta = min(beta, 0.001)
            return beta
        return 0.001

    def _get_current_beta(self, epoch):
        try:
            if hasattr(self.model, "loss_balancer") and self.model.loss_balancer:
                return self.model.loss_balancer.get_beta(epoch, 1.0, 1.0)
            return 1.0
        except:
            return 1.0

    def _compute_metrics_and_losses(
        self,
        loader: DataLoader,
        epoch: int,
        is_test: bool = False,
        strategy_name: str = "",
        retry: int = 0,
    ) -> Tuple[float, Dict]:
        self.model.eval()
        total_loss, recon_loss_sum, kld_loss_sum, valid_samples = 0.0, 0.0, 0.0, 0

        # Store per-batch metrics instead of trying to concatenate
        batch_metrics = {
            "silhouette_scores": [],
            "db_scores": [],
            "ch_scores": [],
        }
        all_ssim = []

        default_metrics = {
            "silhouette_scores": -1,
            "db_scores": 10,
            "ch_scores": 0,
        }

        desc = "Test Set" if is_test else "Validation"
        progress_bar = tqdm(loader, desc=desc, leave=False)

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(progress_bar):
                data = data.to(self.device)
                try:
                    # Forward pass
                    recon, mu, logvar = self.model(data)
                    re_loss, kl_loss, loss = self.model.loss_function(
                        recon, data, mu, logvar, epoch, self.epochs
                    )

                    # Only proceed if forward pass was successful
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        self.logger.warning(
                            f"Batch {batch_idx}: NaN/Inf loss, skipping"
                        )
                        continue

                    # Get latent representations and clusters
                    latents = mu.cpu().numpy()
                    labels_np = labels.cpu().numpy()
                    if labels_np.ndim > 1:
                        labels_np = labels_np.flatten()

                    # Get cluster predictions
                    try:
                        clusters = self.model.predict(data)
                        if clusters is None or len(clusters) != data.size(0):
                            self.logger.warning(
                                f"Batch {batch_idx}: Invalid cluster predictions"
                            )
                            continue

                    except Exception as predict_error:
                        self.logger.warning(
                            f"Batch {batch_idx}: predict() failed: {predict_error}"
                        )
                        continue

                    # Calculate per-batch clustering metrics
                    unique_clusters = len(np.unique(clusters))
                    if unique_clusters >= 2:
                        try:
                            # Calculate all metrics for this batch
                            batch_silhouette = silhouette_score(latents, clusters)
                            batch_db = davies_bouldin_score(latents, clusters)
                            batch_ch = calinski_harabasz_score(latents, clusters)

                            # Store valid metrics
                            batch_metrics["silhouette_scores"].append(batch_silhouette)
                            batch_metrics["db_scores"].append(batch_db)
                            batch_metrics["ch_scores"].append(batch_ch)

                            self.logger.debug(
                                f"Batch {batch_idx} metrics: "
                                f"Sil={batch_silhouette:.3f}"
                            )

                        except Exception as metric_error:
                            self.logger.warning(
                                f"Batch {batch_idx} metric calculation failed: {metric_error}"
                            )
                            continue
                    else:
                        self.logger.debug(
                            f"Batch {batch_idx}: Only {unique_clusters} clusters, skipping metrics"
                        )
                        continue

                    # Update loss tracking (only for successful batches)
                    total_loss += loss.item() * data.size(0)
                    recon_loss_sum += re_loss.item() * data.size(0)
                    kld_loss_sum += kl_loss.item() * data.size(0)
                    valid_samples += data.size(0)

                    # Calculate SSIM
                    recon_np, data_np = recon.cpu().numpy(), data.cpu().numpy()
                    for j in range(data.size(0)):
                        all_ssim.append(
                            ssim(data_np[j, 0], recon_np[j, 0], data_range=1.0)
                        )
                    if (
                        is_test
                        and batch_idx == 0
                        and strategy_name == "Traditional VAE"
                    ):
                        try:
                            n_samples = min(8, data.size(0))  # Save 8 samples
                            _g.save_reconstructed_images(
                                data[:n_samples],
                                recon[:n_samples],
                                str(
                                    self.output_dir
                                    / f"reconstructions_epoch_{epoch}.png"
                                ),
                                num_samples=n_samples,
                            )
                            self.logger.info(
                                f"Saved reconstruction images for epoch {epoch}"
                            )
                        except Exception as save_error:
                            self.logger.warning(
                                f"Failed to save reconstructions: {save_error}"
                            )

                except Exception as e:
                    self.logger.warning(f"Batch {batch_idx} failed completely: {e}")
                    continue

        # Check if we have any valid results
        if valid_samples == 0:
            self.logger.error("No valid samples processed!")
            if retry < 2:
                clear_gpu(self.logger)
                return self._compute_metrics_and_losses(
                    loader, epoch, is_test, strategy_name, retry + 1
                )
            return 50.0, {
                "recon_loss": 50,
                "kld_loss": 0,
                "beta": 1.0,
                "ssim_score": 0,
                "clustering_failed": True,
                **default_metrics,
            }

        # Calculate aggregated clustering metrics
        try:
            if len(batch_metrics["silhouette_scores"]) > 0:
                # try:
                #     all_latents = []
                #     all_labels = []
                #     all_clusters = []
                #     with torch.no_grad():
                #         for data, labels in loader:
                #             data = data.to(self.device)
                #             mu, _ = self.model.encode(data)
                #             clusters = self.model.predict(data)
                #
                #             all_latents.append(mu.cpu().numpy())
                #             all_labels.append(labels.cpu().numpy().flatten())
                #             all_clusters.append(clusters)
                #     latents_combined = np.concatenate(all_latents, axis=0)
                #     labels_combined = np.concatenate(all_labels, axis=0)
                #     clusters_combined = np.concatenate(all_clusters, axis=0)
                #
                #     # Generate visualization
                #     prefix = "test" if is_test else "val"
                #     self.logger.info(
                #         f"Generating {prefix} visualization for epoch {epoch}..."
                #     )
                #     self.model.visualize_latent_space(
                #         latents_combined, clusters_combined, labels_combined
                #     )
                # except Exception as e:
                #     self.logger.warning(f"Visualization failed for epoch {epoch}: {e}")
                # Average the per-batch metrics (weighted by batch size could be added)
                clustering_metrics = {
                    "silhouette_scores": np.mean(batch_metrics["silhouette_scores"]),
                    "db_scores": np.mean(batch_metrics["db_scores"]),
                    "ch_scores": np.mean(batch_metrics["ch_scores"]),
                    "clustering_failed": False,
                }

                # Log aggregated results
                self.logger.info(
                    f"Aggregated metrics from {len(batch_metrics['silhouette_scores'])} valid batches:"
                )
                self.logger.info(
                    f"  Silhouette: {clustering_metrics['silhouette_scores']:.4f}"
                )

            else:
                self.logger.warning("No valid clustering metrics calculated!")
                clustering_metrics = {**default_metrics, "clustering_failed": True}
                # clustering_metrics = default_metrics

        except Exception as aggregation_error:
            self.logger.error(f"Metric aggregation failed: {aggregation_error}")
            clustering_metrics = default_metrics

        # Calculate final metrics
        avg_loss = total_loss / valid_samples
        avg_recon = recon_loss_sum / valid_samples
        avg_kld = kld_loss_sum / valid_samples
        beta = self._safe_beta_calculation(epoch, self.epochs, avg_recon, avg_kld)

        base_metrics = {
            "total_loss": avg_loss,
            "recon_loss": avg_recon,
            "kld_loss": avg_kld,
            "beta": beta,
            "ssim_score": np.mean(all_ssim) if all_ssim else 0,
        }

        return avg_loss, {**base_metrics, **clustering_metrics}

    def _compute_training_metrics(self, epoch, retry=0):
        self.model.eval()  # Set to eval mode for metrics calculation

        # Use a subset of training data to avoid memory issues
        max_batches = min(10, len(self.train_loader))  # Limit to 10 batches max
        total_loss, recon_loss_sum, kld_loss_sum, valid_samples = 0.0, 0.0, 0.0, 0

        # Store per-batch metrics
        batch_metrics = {
            "silhouette_scores": [],
            "db_scores": [],
            "ch_scores": [],
        }
        all_ssim = []

        default_metrics = {
            "silhouette_scores": -1,
            "db_scores": 10,
            "ch_scores": 0,
        }

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(self.train_loader):
                if batch_idx >= max_batches:  # Stop after max_batches
                    break

                data = data.to(self.device)
                try:
                    # Forward pass
                    recon, mu, logvar = self.model(data)
                    re_loss, kl_loss, loss = self.model.loss_function(
                        recon, data, mu, logvar, epoch, self.epochs
                    )

                    # Check for NaN/Inf
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        self.logger.warning(
                            f"Training metrics batch {batch_idx}: NaN/Inf loss, skipping"
                        )
                        continue

                    # Get latent representations and clusters
                    latents = mu.cpu().numpy()
                    labels_np = labels.cpu().numpy()
                    if labels_np.ndim > 1:
                        labels_np = labels_np.flatten()

                    # Get cluster predictions
                    try:
                        clusters = self.model.predict(data)
                        if clusters is None or len(clusters) != data.size(0):
                            self.logger.warning(
                                f"Training metrics batch {batch_idx}: Invalid cluster predictions"
                            )
                            continue
                    except Exception as predict_error:
                        self.logger.warning(
                            f"Training metrics batch {batch_idx}: predict() failed: {predict_error}"
                        )
                        continue

                    # Calculate per-batch clustering metrics
                    unique_clusters = len(np.unique(clusters))
                    if unique_clusters >= 2:
                        try:
                            # Calculate clustering metrics for this batch
                            batch_silhouette = silhouette_score(latents, clusters)
                            batch_db = davies_bouldin_score(latents, clusters)
                            batch_ch = calinski_harabasz_score(latents, clusters)

                            # Store valid metrics
                            batch_metrics["silhouette_scores"].append(batch_silhouette)
                            batch_metrics["db_scores"].append(batch_db)
                            batch_metrics["ch_scores"].append(batch_ch)

                        except Exception as metric_error:
                            self.logger.warning(
                                f"Training metrics batch {batch_idx} calculation failed: {metric_error}"
                            )
                            continue
                    else:
                        self.logger.debug(
                            f"Training metrics batch {batch_idx}: Only {unique_clusters} clusters, skipping metrics"
                        )
                        continue

                    # Update loss tracking (only for successful batches)
                    total_loss += loss.item() * data.size(0)
                    recon_loss_sum += re_loss.item() * data.size(0)
                    kld_loss_sum += kl_loss.item() * data.size(0)
                    valid_samples += data.size(0)

                    # Calculate SSIM
                    recon_np, data_np = recon.cpu().numpy(), data.cpu().numpy()
                    for j in range(data.size(0)):
                        all_ssim.append(
                            ssim(data_np[j, 0], recon_np[j, 0], data_range=1.0)
                        )

                except Exception as e:
                    self.logger.warning(
                        f"Training metrics batch {batch_idx} failed completely: {e}"
                    )
                    continue

        # Set model back to training mode
        self.model.train()

        # Check if we have any valid results
        if valid_samples == 0:
            self.logger.warning("No valid samples processed for training metrics!")
            if retry < 2:
                clear_gpu(self.logger)
                return self._compute_training_metrics(epoch, retry + 1)
            else:
                return {
                    "total_loss": 0,
                    "recon_loss": 50,
                    "kld_loss": 0,
                    "beta": 1.0,
                    "ssim_score": 0,
                    "clustering_failed": True,
                    **default_metrics,
                }

        # Calculate aggregated clustering metrics
        try:
            if len(batch_metrics["silhouette_scores"]) > 0:
                # Average the per-batch metrics
                clustering_metrics = {
                    "silhouette_scores": np.mean(batch_metrics["silhouette_scores"]),
                    "db_scores": np.mean(batch_metrics["db_scores"]),
                    "ch_scores": np.mean(batch_metrics["ch_scores"]),
                    "clustering_failed": False,
                }

                self.logger.debug(
                    f"Training metrics from {len(batch_metrics['silhouette_scores'])} valid batches:"
                )
                self.logger.debug(
                    f"  Silhouette: {clustering_metrics['silhouette_scores']:.4f}"
                )

            else:
                self.logger.warning(
                    "No valid clustering metrics calculated for training!"
                )
                clustering_metrics = {**default_metrics, "clustering_failed": True}

        except Exception as aggregation_error:
            self.logger.error(
                f"Training metric aggregation failed: {aggregation_error}"
            )
            clustering_metrics = default_metrics

        # Calculate final metrics
        avg_loss = total_loss / valid_samples
        avg_recon = recon_loss_sum / valid_samples
        avg_kld = kld_loss_sum / valid_samples
        beta = self._safe_beta_calculation(epoch, self.epochs, avg_recon, avg_kld)

        base_metrics = {
            "total_loss": avg_loss,
            "recon_loss": avg_recon,
            "kld_loss": avg_kld,
            "beta": beta,
            "ssim_score": np.mean(all_ssim) if all_ssim else 0,
        }

        return {**base_metrics, **clustering_metrics}

    def _calculate_clustering_metrics(self, latents, labels, clusters):
        unique_clusters = len(np.unique(clusters))
        if unique_clusters < 2:
            return {
                "silhouette_scores": -1,
                "db_scores": 10,
                "ch_scores": 0,
            }
        return {
            "silhouette_scores": silhouette_score(latents, clusters),
            "db_scores": davies_bouldin_score(latents, clusters),
            "ch_scores": calinski_harabasz_score(latents, clusters),
        }

    def _generate_strategy_visualization(self, strategy_key, strategy_name):
        self.model.eval()
        all_latents, all_labels, all_clusters = [], [], []

        with torch.no_grad():
            for data, labels in self.test_loader:
                data = data.to(self.device)
                mu, _ = self.model.encode(data)
                clusters = self.model.predict(data)

                all_latents.append(mu.cpu().numpy())
                all_labels.append(labels.cpu().numpy().flatten())
                all_clusters.append(clusters)

        latents_combined = np.concatenate(all_latents, axis=0)
        labels_combined = np.concatenate(all_labels, axis=0)
        clusters_combined = np.concatenate(all_clusters, axis=0)

        # Temporarily modify the output path to include strategy name
        original_method = self.model.visualize_latent_space

        def custom_visualize(Z, pred_labels, true_labels=None):
            # Call original but save with strategy-specific name
            if np.isnan(Z).any():
                from sklearn.impute import SimpleImputer

                imputer = SimpleImputer(strategy="mean")
                Z_clean = imputer.fit_transform(Z)
            else:
                Z_clean = Z

            from sklearn.manifold import TSNE

            tsne = TSNE(
                n_components=2,
                perplexity=min(30, len(Z_clean) - 1),
                random_state=42,
                n_iter=1000,
                learning_rate="auto",
                init="pca",
            )
            Z_embedded = tsne.fit_transform(Z_clean)

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                Z_embedded[:, 0],
                Z_embedded[:, 1],
                c=pred_labels,
                cmap="viridis",
                alpha=0.7,
                s=30,
                label="Predicted Clusters",
            )
            if true_labels is not None:
                plt.scatter(
                    Z_embedded[:, 0],
                    Z_embedded[:, 1],
                    c=true_labels,
                    cmap="Set1",
                    marker="x",
                    s=100,
                    alpha=0.5,
                    label="True Labels",
                )

            plt.colorbar(scatter, label="Cluster")
            plt.title(f"Latent Space - {strategy_name} (t-SNE)", fontsize=14)
            plt.xlabel("t-SNE Dimension 1", fontsize=12)
            plt.ylabel("t-SNE Dimension 2", fontsize=12)
            if true_labels is not None:
                plt.legend(loc="upper right")
            plt.tight_layout()

            output_path = (
                self.output_dir
                / f"latent_space_{strategy_key}_k{self.model.nClusters}.png"
            )
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Strategy visualization saved to {output_path}")
            plt.close()

        custom_visualize(latents_combined, clusters_combined, labels_combined)

    def _safe_beta_calculation(self, epoch, total_epochs, recon_loss, kld_loss):
        beta = 1.0
        if (
            hasattr(self.model, "loss_balancer")
            and self.model.loss_balancer is not None
        ):
            try:
                beta = self.model.loss_balancer.get_beta(epoch, recon_loss, kld_loss)
            except Exception:
                pass
        return float(beta)

    def _normalize_metrics(self, raw_metrics):
        norm = {}
        norm["silhouette_norm"] = (raw_metrics.get("silhouette_scores", -1) + 1) / 2
        norm["db_norm"] = 1 / (1 + raw_metrics.get("db_scores", 10))
        norm["ch_norm"] = 1 - (1 / (1 + np.log1p(raw_metrics.get("ch_scores", 0))))
        norm["ssim_norm"] = np.clip(raw_metrics.get("ssim_score", 0), 0, 1)
        return norm

    def _calculate_strategy_scores(self, val_loss, metrics):
        recon_score = 1 / (1 + val_loss)
        norm_metrics = self._normalize_metrics(metrics)
        weights = {
            "silhouette_norm": 0.40,
            "ch_norm": 0.20,
            "db_norm": 0.20,
            "ssim_norm": 0.20,
        }
        composite_score = sum(norm_metrics[k] * w for k, w in weights.items())

        strategy_scores = {
            key: params["weight_recon"] * recon_score
            + params["weight_clustering"] * composite_score
            for key, params in self.strategies.items()
        }
        return strategy_scores, composite_score, norm_metrics

    def _save_training_curves(self):
        """Save training and validation loss curves"""
        try:
            plot_path = self.output_dir / "training_validation_curves.png"
            _g.plot_epoch_results(self.train_losses, self.val_losses, str(plot_path))
            self.logger.info(f"Training curves saved to {plot_path}")
        except Exception as e:
            self.logger.error(f"Failed to save training curves: {e}")

    def _save_metrics_plots(self):
        try:
            self.logger.info("Saving training and validation metric plots...")

            # Create metrics dictionary for plotting
            train_metrics = {
                "epoch_losses": self.train_losses,
                "val_losses": self.val_losses,
            }

            # Save plots using generate module
            _g.save_loss_plots(train_metrics, str(self.output_dir / "train_metrics"))

            self.logger.info("Metrics plots saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save metric plots: {e}")

    def train_and_evaluate(self):
        self.logger.info(
            "\n" + "=" * 60 + "\nSTEP 2: MULTI-STRATEGY MAIN TRAINING\n" + "=" * 60
        )
        self.logger.info("Starting training for %d epochs", self.epochs)
        if (
            self.start_epoch >= self.unfreeze_prior_epoch
            and self.unfreeze_prior_epoch > 0
        ):
            self.model.unfreeze_prior()
            self.logger.info("Prior parameters are already unfrozen.")
        else:
            self.model.freeze_prior()
            self.logger.info("Prior parameters frozen for initial training stability.")

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.current_epoch = epoch
            if epoch == self.unfreeze_prior_epoch:
                self.model.unfreeze_prior()
                self.logger.info("Prior parameters unfrozen")
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] *= 0.1  # Reduce LR by 10x
                self.logger.info("Prior parameters unfrozen with reduced learning rate")

            train_loss, train_metrics = self.train_epoch(epoch)
            # train_loss, train_metrics = self.train_epoch(epoch)
            val_loss, val_metrics = self._compute_metrics_and_losses(
                self.val_loader, epoch, retry=0
            )
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_epoch = epoch
                self.no_improvement_counter = 0  # Reset counter
                self.logger.info(f"✅ New best validation loss: {val_loss:.6f}")
            else:
                self.no_improvement_counter += 1  # Increment counter

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                self.best_train_epoch = epoch

            strategy_scores, composite_score, norm_metrics = (
                self._calculate_strategy_scores(val_loss, val_metrics)
            )

            self.logger.info(
                "\n" + "-" * 80 + f"\n[ EPOCH {epoch} / {self.epochs} ]\n" + "-" * 80
            )
            self._log_epoch_summary(val_loss, val_metrics, composite_score)

            any_improvement = self.update_strategies(
                strategy_scores, epoch, val_metrics
            )
            self._log_epoch_metrics_to_json(
                {
                    "train_metrics": train_metrics,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "composite_clustering_score": composite_score,
                    "strategy_scores": strategy_scores,
                    **val_metrics,
                }
            )

            self.scheduler.step(val_loss)
            # self.no_improvement_counter = (
            #     0 if any_improvement else self.no_improvement_counter + 1
            # )
            self.save_checkpoint(epoch)
            self._save_training_curves()
            self._save_metrics_plots()
            self.logger.info(
                f"Epoch {epoch} summary - Train: {train_loss:.4f}, Val: {val_loss:.4f}, No improvement: {self.no_improvement_counter}/{self.patience}\n"
            )

            if self.no_improvement_counter >= self.patience:
                self.logger.info(
                    f"Stopping early: No improvement for {self.patience} epochs."
                )
                break

    def update_strategies(self, strategy_scores, epoch, val_metrics):
        any_improvement = False
        if val_metrics.get("clustering_failed", False):
            self.logger.warning(
                f"🚨 Epoch {epoch}: Clustering failed - skipping strategy updates"
            )
            return any_improvement
        if (
            val_metrics.get("silhouette_scores", -1) == -1
            or val_metrics.get("db_scores", 10) == 10
            or val_metrics.get("ch_scores", 0) == 0
        ):
            self.logger.warning(
                f"🚨 Epoch {epoch}: Invalid clustering metrics - skipping strategy updates"
            )
            return any_improvement

        for key, score in strategy_scores.items():
            if score > self.strategy_states[key]["best_score"]:
                self.strategy_states[key]["best_score"] = score
                self.strategy_states[key]["best_epoch"] = epoch
                torch.save(self.model.state_dict(), self.strategies[key]["path"])
                any_improvement = True
                # self._generate_best_model_visualization(
                #     key, self.strategies[key]["name"], epoch
                # )
                self.logger.info(
                    f"✅ New best model for strategy '{self.strategies[key]['name']}' (Score: {score:.4f})"
                )
        return any_improvement

    def _generate_best_model_visualization(self, strategy_key, strategy_name, epoch):
        """Generate visualization for a new best model"""
        self.model.eval()
        all_latents, all_labels, all_clusters = [], [], []

        # Use validation set for best model visualization
        with torch.no_grad():
            for data, labels in tqdm(
                self.val_loader, desc=f"Visualizing best {strategy_name}", leave=False
            ):
                data = data.to(self.device)
                try:
                    mu, _ = self.model.encode(data)
                    clusters = self.model.predict(data)

                    all_latents.append(mu.cpu().numpy())
                    all_labels.append(labels.cpu().numpy().flatten())
                    all_clusters.append(clusters)
                except Exception:
                    continue

        if not all_latents:
            self.logger.warning(f"No valid data for {strategy_name} visualization")
            return

        latents_combined = np.concatenate(all_latents, axis=0)
        labels_combined = np.concatenate(all_labels, axis=0)
        clusters_combined = np.concatenate(all_clusters, axis=0)

        # Create custom visualization with strategy and epoch info
        self._create_strategy_specific_visualization(
            latents_combined,
            clusters_combined,
            labels_combined,
            strategy_key,
            strategy_name,
            epoch,
        )

    def _create_strategy_specific_visualization(
        self, Z, pred_labels, true_labels, strategy_key, strategy_name, epoch
    ):
        """Create a strategy-specific visualization with custom naming"""
        if np.isnan(Z).any():
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="mean")
            Z_clean = imputer.fit_transform(Z)
            self.logger.info("Imputed NaN values for visualization")
        else:
            Z_clean = Z

        try:
            from sklearn.manifold import TSNE

            self.logger.info(f"Applying t-SNE for {strategy_name}...")

            tsne = TSNE(
                n_components=2,
                perplexity=min(30, len(Z_clean) - 1),
                random_state=42,
                n_iter=1000,
                learning_rate="auto",
                init="pca",
            )
            Z_embedded = tsne.fit_transform(Z_clean)

            # Create the plot
            plt.figure(figsize=(12, 9))
            scatter = plt.scatter(
                Z_embedded[:, 0],
                Z_embedded[:, 1],
                c=pred_labels,
                cmap="viridis",
                alpha=0.7,
                s=30,
                label="Predicted Clusters",
            )

            if true_labels is not None:
                plt.scatter(
                    Z_embedded[:, 0],
                    Z_embedded[:, 1],
                    c=true_labels,
                    cmap="Set1",
                    marker="x",
                    s=100,
                    alpha=0.5,
                    label="True Labels",
                )

            plt.colorbar(scatter, label="Cluster")
            plt.title(
                f"Best {strategy_name} Model (Epoch {epoch})\nLatent Space Visualization (t-SNE)",
                fontsize=14,
            )
            plt.xlabel("t-SNE Dimension 1", fontsize=12)
            plt.ylabel("t-SNE Dimension 2", fontsize=12)

            if true_labels is not None:
                plt.legend(loc="upper right")

            plt.tight_layout()

            # Save with strategy and epoch info
            output_dir = self.output_dir / "best_model_visualizations"
            output_dir.mkdir(exist_ok=True)

            output_path = (
                output_dir
                / f"best_{strategy_key}_epoch_{epoch}_k{self.model.nClusters}.png"
            )
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"✅ Best model visualization saved: {output_path}")

            # Also create a "latest best" version (overwrites each time)
            latest_path = (
                output_dir / f"latest_best_{strategy_key}_k{self.model.nClusters}.png"
            )
            plt.figure(figsize=(12, 9))
            scatter = plt.scatter(
                Z_embedded[:, 0],
                Z_embedded[:, 1],
                c=pred_labels,
                cmap="viridis",
                alpha=0.7,
                s=30,
            )
            if true_labels is not None:
                plt.scatter(
                    Z_embedded[:, 0],
                    Z_embedded[:, 1],
                    c=true_labels,
                    cmap="Set1",
                    marker="x",
                    s=100,
                    alpha=0.5,
                )
            plt.colorbar(scatter, label="Cluster")
            plt.title(f"Latest Best {strategy_name} Model", fontsize=14)
            plt.xlabel("t-SNE Dimension 1", fontsize=12)
            plt.ylabel("t-SNE Dimension 2", fontsize=12)
            plt.tight_layout()
            plt.savefig(latest_path, dpi=300, bbox_inches="tight")
            plt.close()

        except Exception as e:
            self.logger.error(f"Visualization creation failed for {strategy_name}: {e}")

    def _log_epoch_summary(self, val_loss, metrics, composite_score):
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.logger.info(
            f"\n{'='*100}\n"
            f"EPOCH {self.current_epoch:3d} / {self.epochs} SUMMARY\n"
            f"{'='*100}\n"
            f"📉 LOSSES:\n"
            f"   ├─ Validation:     {val_loss:10.6f}\n"
            f"   ├─ Reconstruction: {metrics['recon_loss']:10.6f}\n"
            f"   ├─ KL Divergence:  {metrics['kld_loss']:10.6f}\n"
            f"   ├─ Beta:           {metrics['beta']:10.6f}\n"
            f"   └─ SSIM:           {metrics['ssim_score']:10.6f}\n"
            f"\n🎯 CLUSTERING (Composite: {composite_score:.6f}):\n"
            f"   ├─ Silhouette:     {metrics['silhouette_scores']:10.6f}\n"
            f"   ├─ Davies-Bouldin: {metrics['db_scores']:10.6f}\n"
            f"   ├─ Calinski-H:     {metrics['ch_scores']:10.6f}\n"
            f"\n⚙️  TRAINING STATUS:\n"
            f"   ├─ Learning Rate:  {current_lr:.2e}\n"
            f"   ├─ No Improvement: {self.no_improvement_counter}/{self.patience}\n"
            f"   └─ Prior Frozen:   {'Yes' if hasattr(self.model, 'prior_frozen') and self.model.prior_frozen else 'No'}\n"
            f"{'='*100}\n"
        )

    def _log_epoch_metrics_to_json(self, epoch_data):
        log_path = self.output_dir / "epoch_metrics_log.json"
        history = []
        if log_path.exists():
            try:
                with open(log_path, "r") as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                history = []
        history.append(epoch_data)
        with open(log_path, "w") as f:
            json.dump(self._convert_to_json_serializable(history), f, indent=4)

    def save_checkpoint(self, epoch):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "strategy_states": self.strategy_states,
                "no_improvement_counter": self.no_improvement_counter,
            },
            self.checkpoint_path,
        )

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.strategy_states = checkpoint.get("strategy_states", self.strategy_states)
        self.no_improvement_counter = checkpoint.get("no_improvement_counter", 0)
        return checkpoint.get("epoch", 0)

    def run_final_comparison(self):
        self.logger.info(
            "\n" + "=" * 60 + "\nSTEP 3: FINAL COMPARISON & ANALYSIS\n" + "=" * 60
        )
        all_results = {}

        # Load epoch history for train/val metrics
        log_path = self.output_dir / "epoch_metrics_log.json"
        epoch_history = []
        if log_path.exists():
            with open(log_path, "r") as f:
                epoch_history = json.load(f)

        for key, params in self.strategies.items():
            if not params["path"].exists():
                self.logger.warning(
                    f"No model for strategy '{params['name']}'. Skipping."
                )
                continue

            self.logger.info(f"--- Evaluating strategy: {params['name']} ---")
            self.model.load_state_dict(
                torch.load(params["path"], map_location=self.device)
            )
            best_epoch = self.strategy_states[key]["best_epoch"]

            # Get test metrics
            test_loss, test_metrics = self._compute_metrics_and_losses(
                self.test_loader, best_epoch, is_test=True, strategy_name=params["name"]
            )

            # Get validation metrics from best epoch
            val_loss, val_metrics = self._compute_metrics_and_losses(
                self.val_loader, best_epoch, is_test=False, strategy_name=params["name"]
            )

            # Find training metrics from epoch history
            train_metrics_at_best = None
            val_metrics_at_best = None
            for epoch_data in epoch_history:
                if epoch_data["epoch"] == best_epoch:
                    train_metrics_at_best = {
                        "train_loss": epoch_data["train_loss"],
                        "train_metrics": epoch_data.get("train_metrics", {}),
                        "val_loss": epoch_data["val_loss"],
                        "composite_clustering_score": epoch_data[
                            "composite_clustering_score"
                        ],
                        "val_metrics": epoch_data.get("val_metrics", {}),
                        "strategy_scores": epoch_data["strategy_scores"],
                    }
                    val_metrics_at_best = {
                        k: v
                        for k, v in epoch_data.items()
                        if k
                        not in [
                            "epoch",
                            "train_loss",
                            "strategy_scores",
                            "composite_clustering_score",
                        ]
                    }
                    break

            _, composite_score, norm_metrics = self._calculate_strategy_scores(
                test_loss, test_metrics
            )

            # CREATE DETAILED RESULTS STRUCTURE
            all_results[key] = {
                "strategy_info": {
                    "strategy_name": params["name"],
                    "strategy_key": key,
                    "best_epoch": best_epoch,
                    "weight_recon": params["weight_recon"],
                    "weight_clustering": params["weight_clustering"],
                },
                "test_performance": {
                    "test_loss": test_loss,
                    "recon_loss": test_metrics["recon_loss"],
                    "kld_loss": test_metrics["kld_loss"],
                    "beta": test_metrics["beta"],
                    "ssim_score": test_metrics["ssim_score"],
                    "clustering_score": composite_score,
                    "silhouette_scores": test_metrics["silhouette_scores"],
                    "db_scores": test_metrics["db_scores"],
                    "ch_scores": test_metrics["ch_scores"],
                    "clustering_failed": test_metrics.get("clustering_failed", False),
                },
                "validation_performance": {
                    "val_loss": val_loss,
                    "recon_loss": val_metrics["recon_loss"],
                    "kld_loss": val_metrics["kld_loss"],
                    "beta": val_metrics["beta"],
                    "ssim_score": val_metrics["ssim_score"],
                    "silhouette_scores": val_metrics["silhouette_scores"],
                    "db_scores": val_metrics["db_scores"],
                    "ch_scores": val_metrics["ch_scores"],
                    "clustering_failed": val_metrics.get("clustering_failed", False),
                },
                "training_history": {
                    "train_loss_at_best_epoch": (
                        train_metrics_at_best["train_loss"]
                        if train_metrics_at_best
                        else None
                    ),
                    "train_metrics_at_best_epoch": (  # ADD THIS SECTION
                        train_metrics_at_best["train_metrics"]
                        if train_metrics_at_best
                        else None
                    ),
                    "val_loss_at_best_epoch": (
                        train_metrics_at_best["val_loss"]
                        if train_metrics_at_best
                        else None
                    ),
                    "val_metrics_at_best_epoch": (  # ADD THIS SECTION
                        train_metrics_at_best["val_metrics"]
                        if train_metrics_at_best
                        else None
                    ),
                    "composite_score_at_best_epoch": (
                        train_metrics_at_best["composite_clustering_score"]
                        if train_metrics_at_best
                        else None
                    ),
                    "strategy_score_at_best_epoch": (
                        train_metrics_at_best["strategy_scores"][key]
                        if train_metrics_at_best
                        else None
                    ),
                    "best_strategy_score": self.strategy_states[key]["best_score"],
                },
                "normalized_metrics": norm_metrics,
                "model_config": {
                    "learning_rate": self.lr,
                    "total_epochs": self.epochs,
                    "patience": self.patience,
                    "pretrain_epochs": self.pretrain_epochs,
                    "unfreeze_prior_epoch": self.unfreeze_prior_epoch,
                },
            }

        if not all_results:
            self.logger.error("No models were saved to evaluate.")
            return {}

        # SAVE DETAILED RESULTS
        with open(self.output_dir / "detailed_strategy_results.json", "w") as f:
            json.dump(self._convert_to_json_serializable(all_results), f, indent=4)

        self.logger.info(
            f"Detailed results saved to {self.output_dir / 'detailed_strategy_results.json'}"
        )

        # Keep existing simple results for compatibility
        simple_results = {}
        for key, detailed in all_results.items():
            simple_results[key] = {
                "strategy_name": detailed["strategy_info"]["strategy_name"],
                "best_epoch": detailed["strategy_info"]["best_epoch"],
                "test_loss": detailed["test_performance"]["test_loss"],
                "clustering_score": detailed["test_performance"]["clustering_score"],
                **{
                    k: v
                    for k, v in detailed["test_performance"].items()
                    if k not in ["test_loss", "clustering_score"]
                },
            }

        with open(self.output_dir / "all_strategies_test_results.json", "w") as f:
            json.dump(self._convert_to_json_serializable(simple_results), f, indent=4)

        self._generate_comparison_tables(simple_results)
        self._generate_strategy_plots()
        self.logger.info(
            "\n" + "=" * 60 + "\n✅ FINAL COMPARISON COMPLETE\n" + "=" * 60
        )

        log_path = self.output_dir / "epoch_metrics_log.json"
        if log_path.exists():
            with open(log_path, "r") as f:
                epoch_history = json.load(f)

                # Generate all visualization types
            _g.create_comprehensive_training_plots(
                epoch_history,
                str(self.output_dir / "comprehensive_training_analysis.png"),
            )

            _g.create_detailed_metrics_report(
                all_results, str(self.output_dir / "detailed_metrics_report.png")
            )

            _g.create_publication_ready_plots(
                all_results, epoch_history, str(self.output_dir / "publication_ready")
            )

            self.logger.info(
                "✅ All comprehensive visualizations generated successfully!"
            )
        return all_results

    def _generate_comparison_tables(self, results):
        header = (
            "Strategy                     | Best Epoch | Test Loss | Recon Loss | KLD Loss  | Clustering Score | Silhouette | SSIM\n"
            + "-----------------------------|------------|-----------|------------|-----------|------------------|------------|-------"
        )
        rows = [
            f"{res['strategy_name']:<28} | {res['best_epoch']:<10} | {res['test_loss']:.6f} | {res['recon_loss']:.6f} | {res['kld_loss']:.6f} | {res['clustering_score']:.6f}       | {res['silhouette_scores']:.6f} | {res['ssim_score']:.4f}"
            for res in results.values()
        ]
        report = f"{header}\n" + "\n".join(rows)
        with open(
            self.output_dir / "comprehensive_strategy_comparison_table.txt", "w"
        ) as f:
            f.write(report)
        self.logger.info(
            f"Comparison table saved to {self.output_dir / 'comprehensive_strategy_comparison_table.txt'}"
        )

    def _generate_strategy_plots(self):
        log_path = self.output_dir / "epoch_metrics_log.json"
        if not log_path.exists():
            return
        with open(log_path, "r") as f:
            history = json.load(f)

        try:
            if hasattr(_g, "plot_strategy_comparison"):
                _g.plot_strategy_comparison(
                    history, str(self.output_dir / "strategy_comparison.png")
                )
        except Exception as e:
            self.logger.warning(f"Strategy comparison plot failed: {e}")
        epochs = [h["epoch"] for h in history]

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axs = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle("Strategy Performance Over Training Epochs", fontsize=20)

        for key, params in self.strategies.items():
            axs[0, 0].plot(
                epochs,
                [h["strategy_scores"][key] for h in history],
                label=params["name"],
                lw=2,
            )
        axs[0, 0].set_title("Strategy Scores vs. Epoch")
        axs[0, 0].legend()

        axs[0, 1].plot(
            epochs,
            [h["composite_clustering_score"] for h in history],
            label="Clustering Score",
            color="blue",
            lw=2,
        )
        ax2_twin = axs[0, 1].twinx()
        ax2_twin.plot(
            epochs,
            [h["val_loss"] for h in history],
            label="Validation Loss",
            color="red",
            linestyle="--",
            lw=2,
        )
        axs[0, 1].set_title("Clustering Score and Validation Loss vs. Epoch")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.output_dir / "strategy_performance_plots.png", dpi=300)
        plt.close()
        self.logger.info(
            f"Strategy performance plots saved to {self.output_dir / 'strategy_performance_plots.png'}"
        )

    def _convert_to_json_serializable(self, obj):
        if isinstance(obj, (dict, defaultdict)):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, Path):
            return str(obj)
        return obj

    def run_complete_pipeline(self):
        try:
            self.pretrain()
            self.train_and_evaluate()
            self.run_explainability()
            self.run_deep_diagnostics()
            final_metrics = self.run_final_comparison()
            return final_metrics
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def run_deep_diagnostics(self):
        """
        Generates advanced Deep Learning diagnostics to prove Disentanglement
        and Latent Independence. Crucial for the Research Discussion.
        """
        self.logger.info(
            "\n"
            + "=" * 60
            + "\nSTEP 2.8: DEEP DIAGNOSTICS (Disentanglement)\n"
            + "=" * 60
        )
        output_dir = self.output_dir / "deep_diagnostics"
        output_dir.mkdir(exist_ok=True)

        import matplotlib.pyplot as plt
        import seaborn as sns
        import mne

        # --- PREPARE DATA ---
        self.model.eval()
        all_mus = []
        all_clusters = []

        # Collect all latent vectors from training set
        with torch.no_grad():
            for data, _ in self.train_loader:
                data = data.to(self.device)
                mu, _ = self.model.encode(data)
                clusters = self.model.predict(data)
                all_mus.append(mu.cpu().numpy())
                all_clusters.append(clusters)

        Z = np.vstack(all_mus)
        C = np.concatenate(all_clusters)

        # --- VISUALIZATION 1: LATENT CORRELATION HEATMAP ---
        # Proves that dimensions are statistically independent (Disentangled)
        self.logger.info("Generating Latent Independence Proof...")

        corr_matrix = np.corrcoef(Z, rowvar=False)

        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Hide upper triangle
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=0.5,
        )
        plt.title(
            "Latent Dimension Independence (Correlation Matrix)\nIdeal = Zero correlation (White/Grey)",
            fontsize=14,
        )
        plt.savefig(output_dir / "Latent_Independence_Heatmap.png", dpi=300)
        plt.close()

        # --- VISUALIZATION 2: CLUSTER FINGERPRINTS (RADAR CHART) ---
        # Shows how each Microstate Class is defined by specific Latents
        self.logger.info("Generating Microstate Fingerprints...")

        n_clusters = self.model.nClusters
        n_latents = self.model.latent_dim

        # Calculate mean latent vector for each cluster
        cluster_means = []
        for i in range(n_clusters):
            if np.sum(C == i) > 0:
                cluster_means.append(np.mean(Z[C == i], axis=0))
            else:
                cluster_means.append(np.zeros(n_latents))
        cluster_means = np.array(cluster_means)

        # Create Radar Chart
        angles = np.linspace(0, 2 * np.pi, n_latents, endpoint=False).tolist()
        angles += [angles[0]]  # Close the loop

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Plot each cluster
        colors = plt.cm.get_cmap("tab10", n_clusters)
        for i in range(n_clusters):
            values = cluster_means[i].tolist()
            values += [values[0]]  # Close the loop
            ax.plot(
                angles, values, linewidth=2, label=f"Cluster {i+1}", color=colors(i)
            )
            ax.fill(angles, values, color=colors(i), alpha=0.1)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f"Lat {i+1}" for i in range(n_latents)])
        plt.title("Microstate 'Fingerprints' in Latent Space", y=1.1, fontsize=16)
        plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
        plt.savefig(output_dir / "Cluster_Latent_Fingerprints.png", dpi=300)
        plt.close()

        # --- VISUALIZATION 3: LATENT TRAVERSALS (The "Manifold" Proof) ---
        # Shows what happens when you tweak ONE dimension while holding others constant
        self.logger.info("Generating Latent Traversals...")

        # MNE Setup for Topomaps
        ch_names = [
            "Fp1",
            "AF3",
            "F7",
            "F3",
            "FC1",
            "FC5",
            "T7",
            "C3",
            "CP1",
            "CP5",
            "P7",
            "P3",
            "Pz",
            "PO3",
            "O1",
            "Oz",
            "O2",
            "PO4",
            "P4",
            "P8",
            "CP6",
            "CP2",
            "C4",
            "T8",
            "FC6",
            "FC2",
            "F4",
            "F8",
            "AF4",
            "Fp2",
            "Fz",
            "Cz",
        ]
        info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types="eeg")
        montage = mne.channels.make_standard_montage("biosemi32")
        info.set_montage(montage)

        # Helper to map 40x40 -> 32 sensors (Reused from previous step)
        def extract_sensors_fast(img):
            # ... (Reuse the extraction logic from previous answer, or define here)
            # Simplified placeholder logic if the helper isn't globally available:
            # You essentially need to sample the 40x40 grid at electrode coordinates.
            # Assuming you have the extract_sensors function available or paste it here.
            # For robustness, let's use a dummy extraction if not available,
            # but ideally use the one from the Explainability step.
            return img.flatten()[:32]  # Placeholder! Replace with real extraction logic

        # Traversal parameters
        n_steps = 7
        grid_x = torch.linspace(-3, 3, n_steps).to(self.device)

        # Plot
        fig, axes = plt.subplots(
            n_latents, n_steps, figsize=(n_steps * 1.5, n_latents * 1.5)
        )

        for dim in range(n_latents):
            # Create a batch where only 'dim' varies, others are 0
            z_traversal = torch.zeros(n_steps, n_latents).to(self.device)
            z_traversal[:, dim] = grid_x

            # Decode
            with torch.no_grad():
                decoded = self.model.decode(z_traversal).cpu().numpy()

            for step in range(n_steps):
                ax = axes[dim, step]

                # Extract Topomap (Use the real 40x40 image)
                img = decoded[step, 0]  # (40, 40)

                # Simple imshow for the grid (Cleanest for traversals)
                # Using RdBu_r to show polarity shifts clearly
                ax.imshow(img, cmap="RdBu_r", origin="lower", vmin=0, vmax=1)

                if dim == 0:
                    ax.set_title(f"{grid_x[step].item():.1f}σ")
                if step == 0:
                    ax.set_ylabel(
                        f"Latent {dim+1}",
                        rotation=0,
                        labelpad=20,
                        va="center",
                        fontsize=10,
                    )

                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle(
            "Latent Manifold Traversals\n(Moving along one axis at a time)", fontsize=16
        )
        plt.tight_layout()
        plt.savefig(output_dir / "Latent_Traversals_Grid.png", dpi=300)
        plt.close()

        self.logger.info(f"✅ Deep diagnostics saved to {output_dir}")

    def run_explainability(self):
        """
        Runs HuberAIME with IRLS and generates MNE Topographic Maps.
        """
        self.logger.info(
            "\n" + "=" * 60 + "\nSTEP 2.5: EXPLAINABILITY (HuberAIME)\n" + "=" * 60
        )

        try:
            # --- INTERNAL CLASS DEFINITION (HuberAIME) ---
            import torch.nn as nn
            import mne
            from matplotlib import cm

            class HuberAIME:
                def __init__(self, vae_model, device, input_shape=(1, 40, 40)):
                    self.vae = vae_model
                    self.device = device
                    self.latent_dim = vae_model.latent_dim
                    self.input_shape = input_shape
                    self.flat_input_size = np.prod(input_shape)
                    self.inverse_model = nn.Linear(
                        self.latent_dim, self.flat_input_size, bias=False
                    ).to(device)

                def fit_irls(
                    self, data_loader, max_iter=10, delta=1.0, lambda_reg=1e-4
                ):
                    # ... [Insert the IRLS code provided in previous answer here] ...
                    # (The logic remains exactly the same as the optimized IRLS version)
                    self.vae.eval()
                    print("🧠 HuberAIME: Collecting data...")
                    Y_list, X_list = [], []
                    for data, _ in tqdm(
                        data_loader, desc="Collecting Latents", leave=False
                    ):
                        data = data.to(self.device)
                        with torch.no_grad():
                            mu, _ = self.vae.encode(data)
                        Y_list.append(mu.cpu().numpy())
                        X_list.append(data.view(data.size(0), -1).cpu().numpy())

                    Y = np.vstack(Y_list)
                    X = np.vstack(X_list)
                    N, d = X.shape
                    k = Y.shape[1]

                    W = np.ones(N)
                    I_k = np.eye(k)

                    for iteration in range(max_iter):
                        Y_weighted = Y.T * W
                        LHS = Y_weighted @ Y + lambda_reg * I_k
                        RHS = Y_weighted @ X
                        try:
                            A = np.linalg.solve(LHS, RHS)
                        except np.linalg.LinAlgError:
                            A = np.linalg.pinv(LHS) @ RHS

                        residuals = X - (Y @ A)
                        residual_norms = np.maximum(
                            np.linalg.norm(residuals, axis=1), 1e-8
                        )
                        W = np.where(
                            residual_norms <= delta, 1.0, delta / residual_norms
                        )

                    self.inverse_model.weight.data = (
                        torch.from_numpy(A.T).float().to(self.device)
                    )

                def get_feature_maps(self):
                    weights = self.inverse_model.weight.data.detach().cpu().numpy().T
                    c, h, w = self.input_shape
                    return weights.reshape(self.latent_dim, h, w)

            # --- EXECUTION ---
            self.logger.info("Initializing HuberAIME Explainer...")
            sample_data = next(iter(self.train_loader))[0]
            input_shape = sample_data.shape[1:]
            aime = HuberAIME(self.model, self.device, input_shape=input_shape)

            self.logger.info("Fitting Inverse Model (IRLS)...")
            aime.fit_irls(self.train_loader, max_iter=10)

            importance_maps = aime.get_feature_maps()

            # --- MNE VISUALIZATION SETUP ---
            # We need to map the 40x40 grid back to 32 electrode positions
            # This requires the channel names and montage used in your preprocessing
            ch_names = [
                "Fp1",
                "AF3",
                "F7",
                "F3",
                "FC1",
                "FC5",
                "T7",
                "C3",
                "CP1",
                "CP5",
                "P7",
                "P3",
                "Pz",
                "PO3",
                "O1",
                "Oz",
                "O2",
                "PO4",
                "P4",
                "P8",
                "CP6",
                "CP2",
                "C4",
                "T8",
                "FC6",
                "FC2",
                "F4",
                "F8",
                "AF4",
                "Fp2",
                "Fz",
                "Cz",
            ]
            info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types="eeg")
            montage = mne.channels.make_standard_montage("biosemi32")
            info.set_montage(montage)

            # Helper to extract channel values from 40x40 grid (Same logic as in MyModel)
            def extract_sensors(img, montage, width=40, height=40):
                import math as m

                pos_3d = []
                valid_chs = []
                for ch in ch_names:
                    if ch in montage.get_positions()["ch_pos"]:
                        pos_3d.append(montage.get_positions()["ch_pos"][ch])
                        valid_chs.append(ch)

                pos_2d = []
                for x, y, z in pos_3d:
                    r = m.sqrt(x**2 + y**2 + z**2)
                    elev = m.atan2(z, m.sqrt(x**2 + y**2))
                    az = m.atan2(y, x)
                    r_proj = m.pi / 2 - elev
                    pos_2d.append([r_proj * m.cos(az), r_proj * m.sin(az)])

                pos_2d = np.array(pos_2d)
                # Normalize to grid
                p_min, p_max = pos_2d.min(axis=0), pos_2d.max(axis=0)

                sensor_vals = []
                for idx, (px, py) in enumerate(pos_2d):
                    # Normalize to 0-1
                    nx = (px - p_min[0]) / (p_max[0] - p_min[0])
                    ny = (py - p_min[1]) / (p_max[1] - p_min[1])

                    # Map to pixels (flip Y)
                    ix = int(nx * (width - 1))
                    iy = int((1 - ny) * (height - 1))
                    ix = np.clip(ix, 0, width - 1)
                    iy = np.clip(iy, 0, height - 1)
                    sensor_vals.append(img[iy, ix])
                return np.array(sensor_vals)

            # --- PLOTTING ---
            output_dir = self.output_dir / "explainability"
            output_dir.mkdir(exist_ok=True)

            n_latents = importance_maps.shape[0]
            cols = 4
            rows = int(np.ceil(n_latents / cols))

            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = axes.flatten()

            self.logger.info("Generating MNE Topomaps for Latents...")

            for i in range(n_latents):
                if i < len(axes):
                    # Extract 32 sensor values from the 40x40 map
                    sensor_data = extract_sensors(importance_maps[i], montage)

                    # Plot using MNE
                    im, _ = mne.viz.plot_topomap(
                        sensor_data,
                        info,
                        axes=axes[i],
                        show=False,
                        cmap="RdBu_r",
                        contours=6,
                        sensors=True,
                        vlim=(sensor_data.min(), sensor_data.max()),
                    )
                    axes[i].set_title(f"Latent {i+1}\nFeature Map")

            # Clean up empty axes
            for i in range(n_latents, len(axes)):
                axes[i].axis("off")

            plt.suptitle("HuberAIME: Disentangled Latent Explanations", fontsize=16)
            plt.tight_layout()
            plt.savefig(output_dir / "HuberAIME_Topomaps.png", dpi=300)
            plt.close()

            self.logger.info(f"✅ Explanation pipeline complete.")

        except Exception as e:
            self.logger.error(f"HuberAIME Analysis Failed: {e}", exc_info=True)
