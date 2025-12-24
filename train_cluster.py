import logging
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader

import model as _m
import clustering_trainer as _t


def train_cluster(
    n_clusters: int,
    config_dict: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    train_set: Any,
    device: torch.device,
    batch_size: int,
    latent_dim: int,
    n_channels: int,
    logger: any = None,
) -> Dict:
    label = f"{n_clusters}_{batch_size}_{latent_dim}"
    base_output_dir = Path(config_dict.get("output_dir", "./outputs"))
    cluster_output_dir = base_output_dir / f"cluster_{label}"
    cluster_output_dir.mkdir(parents=True, exist_ok=True)

    worker_logger = logging.getLogger(f"worker_{label}")
    worker_logger.setLevel(logging.INFO)

    if worker_logger.hasHandlers():
        worker_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    log_file_path = cluster_output_dir / "run.log"
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    worker_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    worker_logger.addHandler(stream_handler)

    worker_logger.propagate = False

    try:
        worker_logger.info(f"Starting training for {label}. Logging to {log_file_path}")

        cluster_config = config_dict.copy()
        cluster_config["output_dir"] = str(cluster_output_dir)

        model = _m.create_model_with_batch_cyclical(
            latent_dim=latent_dim,
            nClusters=n_clusters,
            batch_size=batch_size,
            logger=worker_logger,
            device=device,
            n_cycles_per_epoch=cluster_config.get("n_cycles_per_epoch", 5),
            cycle_ratio=cluster_config.get("cycle_ratio", 0.5),
            gamma=cluster_config.get("gamma", 0.01),
        ).to(device)

        trainer = _t.VAEClusteringTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_set=train_set,
            config=cluster_config,
            device=device,
            cluster_id=n_clusters,
            logger=worker_logger,
        )

        final_metrics = trainer.run_complete_pipeline()

        if not final_metrics:
            raise RuntimeError("Training pipeline did not return final metrics.")

        summary = {
            "n_clusters": n_clusters,
            "batch_size": batch_size,
            "latent_dim": latent_dim,
            "label": label,
            "best_val_loss": trainer.best_val_loss,
            "best_val_epoch": trainer.best_val_epoch,
            "best_train_loss": trainer.best_train_loss,
            "best_train_epoch": trainer.best_train_epoch,
            "final_test_loss": final_metrics.get("test_loss", -1),
            "nmi": final_metrics.get("nmi_scores", -1),
            "ari": final_metrics.get("ari_scores", -1),
            "silhouette": final_metrics.get("silhouette_scores", -1),
            "davies_bouldin": final_metrics.get("db_scores", -1),
            "calinski_harabasz": final_metrics.get("ch_scores", -1),
            "accuracy": final_metrics.get("accuracy", -1),
            "v_measure": final_metrics.get("v_measure_scores", -1),
            "ssim_score": final_metrics.get("ssim_score", -1),
            "batch_success_rate": final_metrics.get("batch_success_rate", 0),
            "sample_success_rate": final_metrics.get("sample_success_rate", 0),
            "train_epochs_completed": len(trainer.train_losses),
            "total_epochs_configured": trainer.epochs,
            "patience": trainer.patience,
            "learning_rate": trainer.lr,
        }

        summary_file = cluster_output_dir / "summary_metrics.yaml"
        with open(summary_file, "w") as f:
            yaml.dump(summary, f, default_flow_style=False)
        worker_logger.info(f"Summary metrics saved to {summary_file}")

        result = {
            "n_clusters": n_clusters,
            "label": label,
            "best_val_loss": trainer.best_val_loss,
            "best_train_loss": trainer.best_train_loss,
            "final_test_metrics": final_metrics,
            "output_dir": str(cluster_output_dir),
            "summary": summary,
            "success": True,
        }

        worker_logger.info(f"Completed training for {label} successfully.")
        return result

    except Exception as e:
        main_logger = logger or logging.getLogger("vae_clustering")
        main_logger.error(f"Error in training cluster {label}: {str(e)}", exc_info=True)
        return {
            "n_clusters": n_clusters,
            "label": label,
            "error": str(e),
            "success": False,
        }
    finally:
        if "worker_logger" in locals():
            for handler in worker_logger.handlers[:]:
                handler.close()
                worker_logger.removeHandler(handler)
