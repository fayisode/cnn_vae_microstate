import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VAE Clustering Training")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--data",
        type=str,
        choices=["anomaly", "house", "eeg"],
        default="eeg",
        help="Dataset to use",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size (overrides config)")
    parser.add_argument(
        "--epochs", type=int, help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--latent_dim", type=int, help="Latent dimension (overrides config)"
    )
    parser.add_argument(
        "--n_clusters", type=int, help="Number of clusters (overrides config)"
    )
    parser.add_argument("--lr", type=float, help="Learning rate (overrides config)")
    parser.add_argument("--seed", type=int, default=20, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Output directory"
    )
    parser.add_argument(
        "--participant",
        type=str,
        default="s01",
        help="Specific participant to train on (e.g., s01, s02, s03, etc.)",
    )

    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="A stable, unique ID for the training session to enable resumption. If not provided, a timestamped directory will be created.",
    )

    return parser.parse_args()
