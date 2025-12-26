import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import torch.multiprocessing as mp
from functools import partial
from typing import Dict, List, Tuple, Any
from itertools import product
from multiprocessing.pool import Pool as ProcessPool

import torch
import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
import helper_function as _g
import process_eeg_signals as _eeg
import seeding as _s
import parse_args as _pa
import train_cluster as _tc
from config.config import config as c

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add this at the top of main() or before any CUDA operations
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# Call immediately

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger("vae_clustering")


# --- NEW: Define Non-Daemonic Process and Pool at the top level ---
class NoDaemonProcess(mp.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(mp.get_context("spawn"))):
    Process = NoDaemonProcess


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(ProcessPool):
    def __init__(self, *args, **kwargs):
        kwargs["context"] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)


# --------------------------------------------------------------------


# If CUDA is available, log device properties
if torch.cuda.is_available():
    x = torch.ones(1).cuda()
else:
    logger.warning("CUDA is not available - PyTorch will use CPU only")


# FIXED: Worker function that creates DataLoaders locally
def _train_worker_function(
    task_id,
    n_clusters,
    batch_size,
    latent_dim,
    config,
    train_dataset,  # ✅ CHANGED: Pass dataset instead of loader
    val_dataset,  # ✅ CHANGED: Pass dataset instead of loader
    test_dataset,  # ✅ CHANGED: Pass dataset instead of loader
    n_channels,
    available_gpu_ids,
    num_gpus,
):
    """Worker function that creates DataLoaders locally to avoid serialization issues."""
    try:
        # Setup worker environment
        # if num_gpus > 0:
        #     gpu_index = task_id % num_gpus
        #     gpu_id = available_gpu_ids[gpu_index]
        #     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        #     device = torch.device("cuda:0")
        #     logger.info(f"Task {task_id} using GPU {gpu_id} (mapped to cuda:0)")
        if num_gpus > 0:
            gpu_index = task_id % num_gpus  # Will cycle between 0 and 1
            device = torch.device(f"cuda:{gpu_index}")
            logger.info(f"Task {task_id} using visible GPU index {gpu_index}")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            device = torch.device("cpu")
            logger.info(f"Task {task_id} using CPU")

        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ✅ CRITICAL FIX: Create DataLoaders in worker process
        from torch.utils.data import DataLoader

        # Conservative DataLoader settings to prevent issues
        logger.info(
            f"Task {task_id}: Creating DataLoaders with batch_size={batch_size}"
        )
        dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": 0,  # Enable parallel loading
        }
        train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)

        logger.info(f"Task {task_id}: DataLoaders created successfully")

        # Validate DataLoaders work
        try:
            # Quick validation - try to get first batch from each loader
            for name, loader in [
                ("train", train_loader),
                ("val", val_loader),
                ("test", test_loader),
            ]:
                data_iter = iter(loader)
                batch = next(data_iter)
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch
                logger.info(
                    f"Task {task_id}: {name} loader validated - batch shape {data.shape}"
                )
        except Exception as e:
            logger.error(f"Task {task_id}: DataLoader validation failed: {e}")
            raise

        # Run training with locally created DataLoaders
        result = _tc.train_cluster(
            n_clusters=n_clusters,
            config_dict=config,
            train_loader=train_loader,  # ✅ Locally created
            val_loader=val_loader,  # ✅ Locally created
            test_loader=test_loader,  # ✅ Locally created
            train_set=train_dataset,  # ✅ Use dataset object
            device=device,
            batch_size=batch_size,
            latent_dim=latent_dim,
            n_channels=n_channels,
            logger=logger,
        )

        logger.info(f"Task {task_id}: Training completed successfully")
        return result

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        logger.error(f"Error in worker {task_id} (n_clusters={n_clusters}): {str(e)}")
        logger.error(f"Traceback: {error_trace}")

        # Cleanup on error
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass

        # Return a minimal result structure to avoid breaking the main process
        return {
            "n_clusters": n_clusters,
            "best_loss": float("inf"),
            "loss_history": None,
            "error": str(e),
        }


class ConfigManager:
    """Handles configuration loading and updates"""

    def __init__(self, args):
        self.args = args
        self.config = c.get_model_config()
        self._update_config_from_args()

    def _update_config_from_args(self):
        """Update config with command-line arguments"""
        updates = {
            "batch_size": self.args.batch_size,
            "epochs": self.args.epochs,
            "latent_dim": self.args.latent_dim,
            "n_clusters": self.args.n_clusters,
            "learning_rate": self.args.lr,
            "output_dir": self.args.output_dir,
        }
        for key, value in updates.items():
            if value is not None:
                self.config[key] = value

    def get_config(self) -> Dict[str, Any]:
        return self.config


class DirectoryManager:
    """Manages output directories and paths"""

    def __init__(self, base_dir: str, participant_id: str = None, run_id: str = None):
        self.base_output_dir = Path(base_dir)
        self.participant_id = participant_id

        if run_id:
            run_name = run_id
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"

        if participant_id:
            self.run_dir = self.base_output_dir / participant_id / run_name
        else:
            self.run_dir = self.base_output_dir / run_name

    def setup_directories(self) -> Path:
        """Create necessary directories"""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def get_comparison_dir(self) -> Path:
        comparison_dir = self.run_dir / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        return comparison_dir


class DataLoaderFactory:
    """Factory for creating data loaders based on data type"""

    @staticmethod
    def get_data_loaders(
        args, config: Dict[str, Any], participant_id: str = None
    ) -> Tuple:
        if args.data == "eeg":
            processor = _eeg.EEGProcessor(c.get_eeg_config(), logger, participant_id)
            datasets = processor.process(
                participant_id=participant_id,
            )
            if hasattr(datasets, "__getitem__") and len(datasets) >= 4:
                return 1, datasets[0], datasets[1], datasets[2], datasets[3]
            else:
                return (
                    1,
                    datasets.train,
                    datasets.validation,
                    datasets.test,
                    datasets.train_set,
                )


class TrainingOrchestrator:
    """Manages the training process"""

    def __init__(
        self, config: Dict[str, Any], run_dir: Path, args, participant_id: str = None
    ):
        self.config = config
        self.run_dir = run_dir
        self.args = args
        self.participant_id = participant_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_gpu_utilization(self):
        """Get GPU utilization information using nvidia-smi"""
        try:
            import subprocess

            command = "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
            output = subprocess.check_output(command.split(), universal_newlines=True)

            gpu_info = []
            for line in output.strip().split("\n"):
                values = [float(x) for x in line.split(", ")]
                gpu_id, gpu_util, mem_used, mem_total = values
                gpu_info.append(
                    {
                        "id": int(gpu_id),
                        "utilization": gpu_util,
                        "memory_used": mem_used,
                        "memory_total": mem_total,
                        "memory_used_percent": (mem_used / mem_total) * 100,
                    }
                )
            return gpu_info
        except Exception as e:
            logger.warning(f"Error getting GPU utilization: {str(e)}")
            return []

    def _get_param_ranges(self) -> List[Tuple[int, int, int]]:
        """Determine ranges for n_channels and latent_dim"""
        cluster_range = [8]
        batch_size_range = [128]
        latent_dim_range = [32]
        param_combinations = list(
            product(cluster_range, batch_size_range, latent_dim_range)
        )

        participant_info = (
            f" for participant {self.participant_id}" if self.participant_id else ""
        )
        logger.info(
            f"Training over {len(param_combinations)} combinations{participant_info}: {param_combinations}"
        )
        return param_combinations

    def train(
        self, train_loader, val_loader, test_loader, train_set, n_channels: int
    ) -> List[Dict]:
        """Execute training process"""
        param_combinations = self._get_param_ranges()

        if len(param_combinations) > 1 and torch.cuda.device_count() > 0:
            return self._train_parallel(
                param_combinations,
                train_loader,
                val_loader,
                test_loader,
                train_set,
                n_channels,
            )
        return self._train_sequential(
            param_combinations,
            train_loader,
            val_loader,
            test_loader,
            train_set,
            n_channels,
        )

    def _train_parallel(self, param_combinations, *args) -> List[Dict]:
        """Execute parallel training with GPU assignment and robust cleanup"""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()  # Will be 2
            available_gpu_ids = list(
                range(num_gpus)
            )  # [0, 1] - these map to physical GPU1,GPU2
            logger.info(f"Using {num_gpus} GPUs with indices: {available_gpu_ids}")
        else:
            num_gpus = 0
            available_gpu_ids = []
            logger.warning("No GPUs available, using CPU")

        # remove
        processes_per_gpu = 3
        n_processes = min(num_gpus * processes_per_gpu, len(param_combinations))

        participant_info = (
            f" for participant {self.participant_id}" if self.participant_id else ""
        )
        logger.info(
            f"Starting parallel training with {n_processes} processes{participant_info}"
        )

        train_loader, val_loader, test_loader, train_set, n_channels = args

        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        test_dataset = test_loader.dataset

        logger.info("✅ Extracted datasets for worker processes")

        if torch.cuda.is_available():
            torch.cuda.init()
            if available_gpu_ids:
                torch.cuda.set_device(available_gpu_ids[0])

        mp.set_start_method("spawn", force=True)

        # Use the MyPool class defined at the top level of the script
        pool = MyPool(processes=n_processes)
        results = []

        try:
            # This list will hold results from completed jobs found on disk
            final_results = []
            # This list will hold async jobs to be executed
            async_results = []

            for i, (n_clusters, batch_size, latent_dim) in enumerate(
                param_combinations
            ):

                label = f"{n_clusters}_{batch_size}_{latent_dim}"
                expected_output_dir = self.run_dir / f"cluster_{label}"
                success_marker = expected_output_dir / "summary_metrics.yaml"

                if success_marker.exists():
                    logger.info(
                        f"Skipping task {i} for {label} as it has already completed."
                    )
                    try:
                        with open(success_marker, "r") as f:
                            summary = yaml.safe_load(f)

                        final_results.append(
                            {
                                "n_clusters": summary.get("n_clusters"),
                                "best_loss": summary.get("best_loss"),
                                "loss_history": {
                                    "nmi_scores": [summary.get("nmi")],
                                    "ari_scores": [summary.get("ari")],
                                    "silhouette_scores": [summary.get("silhouette")],
                                    "db_scores": [summary.get("davies_bouldin")],
                                    "ch_scores": [summary.get("calinski_harabasz")],
                                },
                                "output_dir": str(expected_output_dir),
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not load summary for completed task {label}: {e}"
                        )
                    continue

                logger.info(
                    f"Scheduling task {i}{participant_info}: clusters={n_clusters}, batch_size={batch_size}, latent_dim={latent_dim}"
                )

                # ✅ CRITICAL FIX: Pass datasets instead of DataLoaders
                result = pool.apply_async(
                    _train_worker_function,
                    args=(
                        i,
                        n_clusters,
                        batch_size,
                        latent_dim,
                        self.config,
                        train_dataset,  # ✅ CHANGED: Pass dataset
                        val_dataset,  # ✅ CHANGED: Pass dataset
                        test_dataset,  # ✅ CHANGED: Pass dataset
                        n_channels,
                        available_gpu_ids,
                        num_gpus,
                    ),
                )
                async_results.append((i, result))

            task_timeout = 3600 * 24  # 24 hour timeout per task

            for i, result in async_results:
                try:
                    task_result = result.get(timeout=task_timeout)
                    final_results.append(task_result)
                    logger.info(f"Task {i}{participant_info} completed successfully")
                except mp.TimeoutError:
                    logger.error(
                        f"Task {i}{participant_info} timed out after {task_timeout} seconds"
                    )
                    final_results.append(
                        {
                            "n_clusters": param_combinations[i][0],
                            "best_loss": float("inf"),
                            "loss_history": None,
                            "error": f"Timeout after {task_timeout} seconds",
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"Task {i}{participant_info} failed with error: {str(e)}"
                    )
                    import traceback

                    logger.error(f"Traceback: {traceback.format_exc()}")
                    final_results.append(
                        {
                            "n_clusters": param_combinations[i][0],
                            "best_loss": float("inf"),
                            "loss_history": None,
                            "error": str(e),
                        }
                    )

            return final_results

        except KeyboardInterrupt:
            logger.warning("Received KeyboardInterrupt, terminating workers")
            pool.terminate()
            return []
        except Exception as e:
            logger.error(f"Error in parallel training orchestration: {str(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
        finally:
            logger.info("Cleaning up process pool")
            try:
                pool.close()
                pool.join()
            except Exception as e:
                logger.error(f"Error during pool cleanup: {str(e)}")
                pool.terminate()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    import gc

                    gc.collect()
                    logger.info("CUDA resources and memory cleaned up")
                except Exception as e:
                    logger.error(f"Error cleaning CUDA resources: {str(e)}")

    def _train_sequential(self, param_combinations, *args) -> List[Dict]:
        """Execute sequential training"""
        participant_info = (
            f" for participant {self.participant_id}" if self.participant_id else ""
        )
        logger.info(
            f"Running sequential training for parameter combinations{participant_info}"
        )
        results = []

        train_loader, val_loader, test_loader, train_set, n_channels = args

        for n_clusters, batch_size, latent_dim in param_combinations:
            result = _tc.train_cluster(
                n_clusters=n_clusters,
                config_dict=self.config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                train_set=train_set,
                device=self.device,
                batch_size=batch_size,
                latent_dim=latent_dim,
                n_channels=n_channels,
                logger=logger,
            )
            results.append(result)
        return results


class ResultsProcessor:
    """Processes and saves training results"""

    @staticmethod
    def process_results(
        results: List[Dict],
        comparison_dir: Path,
        run_dir: Path,
        participant_id: str = None,
    ) -> None:
        """Process training results and generate summaries"""
        if not results:
            logger.error("No results to process")
            return

        summary_data = []
        loss_history_mean = {
            "kld_losses": [],
            "reconstruct_losses": [],
            "epoch_losses": [],
            "nmi_scores": [],
            "ari_scores": [],
            "beta_scores": [],
            "silhouette_scores": [],
            "db_scores": [],
            "ch_scores": [],
        }

        for result in results:
            if result and result.get("loss_history") is not None:
                summary_data.append(ResultsProcessor._extract_summary(result))
                ResultsProcessor._update_loss_history(
                    loss_history_mean, result["loss_history"]
                )
            else:
                participant_info = (
                    f" for participant {participant_id}" if participant_id else ""
                )
                cluster_num = result.get("n_clusters", "N/A") if result else "N/A"
                logger.warning(
                    f"Skipping results for {cluster_num} clusters{participant_info} due to error or no history"
                )

        if summary_data:
            ResultsProcessor._save_results(
                loss_history_mean, comparison_dir, summary_data, run_dir, participant_id
            )
        else:
            logger.error("No valid results to save")

    @staticmethod
    def _extract_summary(result: Dict) -> Dict:
        """Extract summary data from a single result"""
        metrics = [
            "nmi_scores",
            "ari_scores",
            "silhouette_scores",
            "db_scores",
            "ch_scores",
        ]
        summary = {"n_clusters": result["n_clusters"], "best_loss": result["best_loss"]}
        for metric in metrics:
            scores = result["loss_history"].get(metric, [])
            summary[metric.replace("_scores", "")] = scores[-1] if scores else None
        summary["output_dir"] = result.get("output_dir", "")
        return summary

    @staticmethod
    def _update_loss_history(loss_history_mean: Dict, loss_history: Dict) -> None:
        """Update mean loss history"""
        for key in loss_history_mean:
            if key in loss_history and loss_history[key]:
                value = (
                    np.mean(loss_history[key])
                    if isinstance(loss_history[key], (list, np.ndarray))
                    else loss_history[key]
                )
                loss_history_mean[key].append(value)

    @staticmethod
    def _save_results(
        loss_history_mean: Dict,
        comparison_dir: Path,
        summary_data: List[Dict],
        run_dir: Path,
        participant_id: str = None,
    ) -> None:
        """Save all results and generate plots"""
        try:
            filename_prefix = f"loss_history_mean_all_cluster"
            if participant_id:
                filename_prefix = f"loss_history_mean_all_cluster_{participant_id}"

            _g.save_loss_plots(loss_history_mean, str(comparison_dir / filename_prefix))
            logger.info(f"Saved combined loss plots to {comparison_dir}")

            if len(summary_data) > 1:
                ResultsAnalyzer.analyze_and_save(
                    summary_data, comparison_dir, run_dir, participant_id
                )
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")


class ResultsAnalyzer:
    """Analyzes results and generates comparison metrics"""

    @staticmethod
    def analyze_and_save(
        summary_data: List[Dict],
        comparison_dir: Path,
        run_dir: Path,
        participant_id: str = None,
    ) -> None:
        """Analyze results and save comparisons"""
        df = pd.DataFrame(summary_data).sort_values("n_clusters")
        ResultsAnalyzer._save_summary_files(df, comparison_dir, participant_id)
        ResultsAnalyzer._generate_plots(df, comparison_dir, participant_id)
        ResultsAnalyzer._save_best_configurations(
            df, comparison_dir, run_dir, participant_id
        )

    @staticmethod
    def _save_summary_files(
        df: pd.DataFrame, comparison_dir: Path, participant_id: str = None
    ) -> None:
        """Save summary data in CSV and YAML formats"""
        csv_name = "cluster_comparison.csv"
        yaml_name = "cluster_comparison.yaml"
        if participant_id:
            csv_name = f"cluster_comparison_{participant_id}.csv"
            yaml_name = f"cluster_comparison_{participant_id}.yaml"

        df.to_csv(comparison_dir / csv_name, index=False)
        with open(comparison_dir / yaml_name, "w") as f:
            yaml.dump(df.to_dict("records"), f, default_flow_style=False)

    @staticmethod
    def _generate_plots(
        df: pd.DataFrame, comparison_dir: Path, participant_id: str = None
    ) -> None:
        """Generate comparison plots"""
        plt.figure(figsize=(12, 6))
        plt.plot(df["n_clusters"], df["best_loss"], "o-", linewidth=2)
        plt.xlabel("Number of Clusters")
        plt.ylabel("Validation Loss")

        title = "Validation Loss vs Number of Clusters"
        if participant_id:
            title = f"Validation Loss vs Number of Clusters - {participant_id}"
        plt.title(title)

        plt.grid(True)

        plot_name = "validation_loss_by_clusters.png"
        if participant_id:
            plot_name = f"validation_loss_by_clusters_{participant_id}.png"

        plt.savefig(comparison_dir / plot_name)
        plt.close()

    @staticmethod
    def _save_best_configurations(
        df: pd.DataFrame,
        comparison_dir: Path,
        run_dir: Path,
        participant_id: str = None,
    ) -> None:
        """Save best configurations and create model links"""
        if df.empty or "best_loss" not in df.columns or df["best_loss"].isnull().all():
            logger.warning(
                "DataFrame is empty or has no valid losses; cannot determine best configuration."
            )
            return

        best_metrics = {
            "best_by_validation_loss": df.loc[df["best_loss"].idxmin()].to_dict()
        }

        if participant_id:
            best_metrics["participant_id"] = participant_id

        config_name = "best_configurations.yaml"
        if participant_id:
            config_name = f"best_configurations_{participant_id}.yaml"

        with open(comparison_dir / config_name, "w") as f:
            yaml.dump(best_metrics, f, default_flow_style=False)


def main():
    """Main execution function"""

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    try:
        mp.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    try:
        mp.set_start_method("spawn", force=True)

        try:
            args = _pa.parse_args()
        except Exception as e:
            logger.error(f"Error parsing arguments: {str(e)}")
            return

        try:
            _s.set_seed(args.seed)
        except Exception as e:
            logger.error(f"Error setting seed: {str(e)}")
            return

        try:
            config_manager = ConfigManager(args)
            config = config_manager.get_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return

        participant_id = args.participant
        if participant_id:
            logger.info(f"Training configured for participant: {participant_id}")

        try:
            dir_manager = DirectoryManager(
                config.get("output_dir", "./outputs"), participant_id, args.run_id
            )
            run_dir = dir_manager.setup_directories()
            config["output_dir"] = str(run_dir)

            with open(run_dir / "config.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Error setting up directories or saving config: {str(e)}")
            return

        logger.info(f"Starting new run in directory: {run_dir}")

        try:
            n_channels, train_loader, val_loader, test_loader, train_set = (
                DataLoaderFactory.get_data_loaders(args, config, participant_id)
            )
        except Exception as e:
            logger.error(f"Error creating data loaders: {str(e)}")
            return

        try:
            orchestrator = TrainingOrchestrator(config, run_dir, args, participant_id)
            results = orchestrator.train(
                train_loader, val_loader, test_loader, train_set, n_channels
            )
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return

        try:
            ResultsProcessor.process_results(
                results, dir_manager.get_comparison_dir(), run_dir, participant_id
            )
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            return

        participant_info = (
            f" for participant {participant_id}" if participant_id else ""
        )
        logger.info(f"Training pipeline completed successfully{participant_info}")

    except Exception as e:
        logger.critical(f"Unexpected error in main execution: {str(e)}")
        import traceback

        logger.critical(traceback.format_exc())


if __name__ == "__main__":
    main()
