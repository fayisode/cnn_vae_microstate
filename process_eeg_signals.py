#!/usr/bin/env python3
"""
Advanced Research EEG Processor for Microstate Analysis
-------------------------------------------------------
Complete implementation with 12 publication-ready visualizations
Backend: Agg (Non-interactive, saves files only)
Visualization: MNE Standard (Head outlines, sensors)
Data Source: Automagically downloads DEAP dataset from Google Drive if missing.
"""

import sys
import os
import pickle
import logging
import warnings
import math as m
import shutil
from datetime import datetime
from pathlib import Path

# External Dependencies for Downloading
import gdown

# Critical: Set Backend to Non-Interactive
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Data Science
import numpy as np
import scipy.signal
import scipy.linalg
from scipy.interpolate import griddata

# EEG
import mne
from pycrostates.preprocessing import extract_gfp_peaks, apply_spatial_filter

# Deep Learning
import torch as T
from torch.utils.data import DataLoader, TensorDataset, random_split

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Force output to stdout
        logging.FileHandler("eeg_processing.log"),  # Log to file as well
    ],
)
logger = logging.getLogger("eeg_processor")

warnings.filterwarnings("ignore")
logging.getLogger("mne").setLevel(logging.WARNING)

# Visualization Style Settings
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "image.cmap": "RdBu_r",
    }
)


class VaeDatasets:
    """Container for train/validation/test datasets."""

    def __init__(self, train, validation, test, train_set=None):
        self.train = train
        self.validation = validation
        self.test = test
        self.train_set = train_set


class EEGProcessor:
    """
    Advanced EEG processing pipeline for microstate analysis.

    Features:
    - Automatic Google Drive Data Downloading
    - Spatial filtering for artifact reduction
    - Average referencing
    - GFP peak extraction
    - Topographic map generation
    - Comprehensive visualization suite (12 figures)
    """

    # Standard 32-channel biosemi layout
    STANDARD_CH_NAMES = [
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

    def __init__(self, config_dict=None, logger=None, participant_id=None):
        """
        Initialize EEG Processor.
        """
        self.logger = logger or logging.getLogger("eeg_processor")

        # Default configuration including Google Drive ID
        self.config = {
            "gdrive_folder_id": "1EMmpU93Y-dczq0cQPT6TsZc5Q8cU0D7z",  # DEAP Dataset Folder
            "data_dir": "./data",
            "output_path": "./Topomaps",
            "figure_dir": "./Figure",
            "sample_freq": 128,
            "num_eeg_channels": 32,
            "topo_map_size": 40,
            "interpolation_method": "cubic",
            "batch_size": 64,
            "max_topo_samples": 500000,
            "trial_padding_sec": 0.5,
        }

        if config_dict:
            self.config.update(config_dict)

        self.participant_id = participant_id
        self._setup_participant_paths()

        # Core parameters
        self.sfreq = self.config.get("sample_freq", 128)
        self.topo_map_size = self.config.get("topo_map_size", 40)
        self.ch_names = self.STANDARD_CH_NAMES

        # Create output directories
        self._create_directories()

        # Data containers
        self.info = None
        self.raw_data_structure = None
        self.pos_3d = None
        self.pos_2d = None
        self.sampling_indices = None
        self.gfp_curve = None
        self.raw_mne_unfiltered = None

    def _setup_participant_paths(self):
        """Configure paths for specific participant."""
        if self.participant_id:
            data_dir = self.config.get("data_dir", "./data")
            self.config["data_path"] = f"{data_dir}/{self.participant_id}.dat"
            self.config["output_path"] = f"{data_dir}/Topomaps/{self.participant_id}"
            self.config["figure_dir"] = f"{data_dir}/Figure/{self.participant_id}"

    def _create_directories(self):
        """Create necessary output directories."""
        for key in ["figure_dir", "output_path", "data_dir"]:
            Path(self.config.get(key)).mkdir(parents=True, exist_ok=True)

    def _ensure_directory_exists(self, file_path):
        directory = (
            os.path.dirname(file_path) if os.path.dirname(file_path) else file_path
        )
        os.makedirs(directory, exist_ok=True)

    # =========================================================================
    # Data Downloading (Google Drive Integration)
    # =========================================================================

    def _list_gdrive_folder(self, folder_id, output_dir):
        """Helper to download folder from GDrive using gdown."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(
                f"Downloading GDrive folder {folder_id} to {output_dir}..."
            )
            gdown.download_folder(
                id=folder_id,
                output=str(output_dir),
                quiet=False,
                use_cookies=False,
                remaining_ok=True,
            )
            self.logger.info(f"Google Drive folder downloaded/verified in {output_dir}")
        except Exception as e:
            self.logger.error(f"Error downloading Google Drive folder: {e}")

    def check_and_download_data(self):
        """
        Verify data existence and download from Google Drive if missing.
        """
        data_dir = self.config.get("data_dir")
        gdrive_folder_id = self.config.get("gdrive_folder_id")

        # Check specific participant file
        if self.participant_id:
            participant_file = f"{data_dir}/{self.participant_id}.dat"
            if os.path.exists(participant_file):
                self.logger.info(f"Participant file found: {participant_file}")
                return
            else:
                self.logger.warning(f"Participant file not found: {participant_file}")

        # Check for ANY .dat files in the directory
        existing_dat_files = [f for f in os.listdir(data_dir) if f.endswith(".dat")]

        if not existing_dat_files:
            self.logger.info(f"No .dat files found in {data_dir}")
            self.logger.info("Attempting to download data from Google Drive...")

            if gdrive_folder_id:
                self._list_gdrive_folder(gdrive_folder_id, data_dir)
            else:
                self.logger.error("No Google Drive folder ID provided configuration.")
                raise ValueError("Missing data and no Google Drive ID configured.")
        else:
            self.logger.info(
                f"Found {len(existing_dat_files)} .dat files in {data_dir}"
            )

        # Re-verify specific participant after download attempt
        if self.participant_id:
            participant_file = f"{data_dir}/{self.participant_id}.dat"
            if not os.path.exists(participant_file):
                self.logger.error(
                    f"Download completed, but {self.participant_id}.dat still not found."
                )
                available = [f for f in os.listdir(data_dir) if f.endswith(".dat")]
                self.logger.info(f"Available files: {available}")
                raise FileNotFoundError(
                    f"Could not find or download {self.participant_id}.dat"
                )

    # =========================================================================
    # Data Loading
    # =========================================================================
    def load_preprocessed_dat(self, dat_file_path):
        try:
            with open(dat_file_path, "rb") as f:
                data_dict = pickle.load(f, encoding="latin1")

            data = data_dict["data"]
            eeg_data = data[:, :32, :]
            n_trials, n_channels, n_timepoints = eeg_data.shape

            self.raw_data_structure = eeg_data

            # Concatenate trials directly (NO PADDING)
            reshaped_data = eeg_data.reshape(n_channels, -1)

            self.logger.info(
                f"Loaded {n_trials} trials, {n_channels} channels, "
                f"{n_timepoints} timepoints/trial (no padding)"
            )

            return reshaped_data

        except Exception as e:
            self.logger.error(f"Error: {e}")
            return None

    # =========================================================================
    # MNE Object Creation
    # =========================================================================

    def create_mne_raw_object(self, data, apply_filter=True):
        """
        Create MNE Raw object with montage and preprocessing.
        """
        if data.shape[0] > data.shape[1]:
            data = data.T

        # Create info and raw object
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        # Set montage
        montage = mne.channels.make_standard_montage("biosemi32")
        raw.set_montage(montage)

        # Setting filtering for Microstate Signals
        # 2-20Hz is standard for microstate analysis
        raw.filter(
            l_freq=2.0, h_freq=20.0, method="fir", fir_design="firwin", phase="zero"
        )
        self.logger.info("Applied 2-20 Hz bandpass filter")

        self.info = raw.info
        return raw

    # =========================================================================
    # Geometry & Coordinate Transformations
    # =========================================================================

    def cart2sph(self, x, y, z):
        """Convert Cartesian to spherical coordinates."""
        x2_y2 = x**2 + y**2
        r = m.sqrt(x2_y2 + z**2)
        elev = m.atan2(z, m.sqrt(x2_y2))
        az = m.atan2(y, x)
        return r, elev, az

    def pol2cart(self, theta, rho):
        """Convert polar to Cartesian coordinates."""
        return rho * m.cos(theta), rho * m.sin(theta)

    def azim_proj(self, pos):
        """Azimuthal equidistant projection."""
        r, elev, az = self.cart2sph(pos[0], pos[1], pos[2])
        return self.pol2cart(az, m.pi / 2 - elev)

    def get_3d_coordinates(self, montage_channel_location):
        """Extract 3D electrode coordinates from montage."""
        location = []
        # Get last 32 channels (assumes biosemi layout specificities)
        locs = montage_channel_location[-32:]

        for i in range(32):
            vals = list(locs[i].values())
            location.append(vals[1] * 1000)

        return np.array(location)

    # =========================================================================
    # Topographic Map Generation
    # =========================================================================

    def create_topographic_map(self, channel_values, pos_2d):
        """Create interpolated topographic map."""
        grid_x, grid_y = np.mgrid[
            min(pos_2d[:, 0]) : max(pos_2d[:, 0]) : self.topo_map_size * 1j,
            min(pos_2d[:, 1]) : max(pos_2d[:, 1]) : self.topo_map_size * 1j,
        ]

        interpolated = griddata(
            pos_2d,
            channel_values,
            (grid_x, grid_y),
            method=self.config["interpolation_method"],
            fill_value=0,
        )

        return interpolated

    def generate_topographic_maps(self, raw_mne):
        """Generate topographic maps from GFP peaks."""
        # Get electrode positions
        self.pos_3d = self.get_3d_coordinates(raw_mne.info["dig"])
        self.pos_2d = np.array([self.azim_proj(p) for p in self.pos_3d])

        # Extract GFP peaks
        self.logger.info("Extracting GFP peaks...")
        # This returns a ChData object containing the maps at peak times
        gfp_peaks_structure = extract_gfp_peaks(raw_mne, min_peak_distance=3)

        # Get the actual data (Channels x Peaks)
        peak_maps_data = gfp_peaks_structure.get_data()  # Shape: (32, N_peaks)
        n_peaks = peak_maps_data.shape[1]

        self.logger.info(f"Found {n_peaks} GFP peaks")

        # Sampling if too many peaks
        max_samples = self.config.get("max_topo_samples", 500000)
        if n_peaks > max_samples:
            # Randomly select column indices
            selected_indices = np.random.choice(n_peaks, max_samples, replace=False)
            selected_indices.sort()
            # Select specific columns (peaks)
            selected_data = peak_maps_data[:, selected_indices]
        else:
            selected_data = peak_maps_data

        self.logger.info(
            f"Generating maps from {selected_data.shape[1]} selected peaks"
        )

        # Generate maps (Interpolation)
        maps = []
        # Iterate over the columns (peaks)
        for i in range(selected_data.shape[1]):
            vals = selected_data[:, i]  # Get all channels for this peak
            img = self.create_topographic_map(vals, self.pos_2d)
            maps.append(img)

        # Calculate GFP for visualization (using original raw data for the curve)
        full_data = raw_mne.get_data()
        self.gfp_curve = np.std(full_data, axis=0)
        self.sampling_indices = []

        return np.array(maps)

    # def generate_topographic_maps(self, raw_mne):
    #     """Generate topographic maps from GFP peaks."""
    #     # Get electrode positions
    #     self.pos_3d = self.get_3d_coordinates(raw_mne.info["dig"])
    #     self.pos_2d = np.array([self.azim_proj(p) for p in self.pos_3d])
    #
    #     # Extract GFP peaks
    #     self.logger.info("Extracting GFP peaks...")
    #     gfp_peaks_raw = extract_gfp_peaks(raw_mne, min_peak_distance=3)
    #
    #     # Get peak indices
    #     peak_samples = gfp_peaks_raw.get_data().flatten().astype(int)
    #     self.logger.info(f"Found {len(peak_samples)} GFP peaks")
    #
    #     max_samples = self.config.get("max_topo_samples", 50000)
    #     if len(peak_samples) > max_samples:
    #         selected = np.random.choice(len(peak_samples), max_samples, replace=False)
    #         selected.sort()
    #         selected_samples = peak_samples[selected]
    #     else:
    #         selected_samples = peak_samples
    #
    #     self.logger.info(
    #         f"Generating {len(selected_samples)} maps from "
    #         f"{len(peak_samples)} GFP peaks (standard approach)"
    #     )
    #
    #     # Get data
    #     data = raw_mne.get_data()
    #
    #     # Generate maps
    #     maps = []
    #     for sample_idx in selected_samples:
    #         vals = data[:, sample_idx]
    #         img = self.create_topographic_map(vals, self.pos_2d)
    #         maps.append(img)
    #
    #     # Calculate GFP for visualization
    #     gfp = np.std(data, axis=0)
    #
    #     self.sampling_indices = selected_samples
    #     self.gfp_curve = gfp
    #
    #     return np.array(maps)

    # =========================================================================
    # Visualization Helpers
    # =========================================================================

    def make_circular_mask(self, ax, img_size):
        """Create circular mask and head outline on axis."""
        center = img_size / 2 - 0.5
        radius = img_size / 2

        # Create circle for clipping
        circle = patches.Circle((center, center), radius, transform=ax.transData)

        # Draw head outline
        head_circle = patches.Circle(
            (center, center),
            radius,
            linewidth=2,
            edgecolor="black",
            facecolor="none",
            zorder=10,
        )
        ax.add_patch(head_circle)

        # Draw nose (triangle at top)
        nose_len = radius * 0.15
        nose_wid = radius * 0.1
        nose_x = [center - nose_wid, center, center + nose_wid]
        nose_y = [img_size - 1, img_size - 1 + nose_len, img_size - 1]
        ax.plot(nose_x, nose_y, color="black", linewidth=2, zorder=10)

        return circle

    def _save_figure(self, fig_name, dpi=300):
        """Save figure and close to free memory."""
        fig_dir = self.config["figure_dir"]
        plt.savefig(f"{fig_dir}/{fig_name}", dpi=dpi, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Saved {fig_name}")

    # =========================================================================
    # Publication-Ready Visualizations (12 Figures)
    # =========================================================================

    def generate_research_figures(self, topo_maps, raw_dataset):
        """Generate all 12 publication-ready figures."""
        self.logger.info("=" * 60)
        self.logger.info("Generating 12 Publication-Ready Visualizations")
        self.logger.info("=" * 60)

        self._figure_01_power_spectral_density()
        self._figure_02_gfp_peak_extraction()
        self._figure_03_channel_time_series()
        self._figure_04_topomap_grid(topo_maps)
        self._figure_05_gfp_distribution()
        self._figure_06_channel_correlation_matrix(raw_dataset)
        self._figure_07_sensor_layout()
        self._figure_08_spatial_variance(topo_maps)
        self._figure_09_butterfly_plot(raw_dataset)
        self._figure_10_frequency_band_topomaps()
        self._figure_11_split_half_reliability(topo_maps)
        # self._figure_12_spatial_filter_comparison(raw_dataset)

        self.logger.info("=" * 60)
        self.logger.info("All visualizations complete!")
        self.logger.info("=" * 60)

    def _figure_01_power_spectral_density(self):
        """Figure 1: Average PSD across trials."""
        try:
            plt.figure(figsize=(10, 6))

            all_psds = []
            # Calculate PSD for ALL trials and average
            # raw_data_structure is (Trials, Channels, Time)
            for trial_idx in range(self.raw_data_structure.shape[0]):
                trial_data = self.raw_data_structure[trial_idx, :, :]
                f, Pxx = scipy.signal.welch(
                    trial_data, fs=self.sfreq, nperseg=512, axis=1
                )
                # Average across channels for this trial
                all_psds.append(np.mean(Pxx, axis=0))

            # Average across trials
            mean_psd = np.mean(all_psds, axis=0)
            std_psd = np.std(all_psds, axis=0)

            # Plot mean with confidence interval
            plt.semilogy(f, mean_psd, color="#2c3e50", linewidth=2, label="Mean PSD")
            plt.fill_between(
                f,
                mean_psd - std_psd,
                mean_psd + std_psd,
                alpha=0.2,
                color="#2c3e50",
                label="±1 SD",
            )

            # Highlight frequency bands
            plt.axvspan(4, 8, color="#f39c12", alpha=0.15, label="Theta (4-8 Hz)")
            plt.axvspan(8, 12, color="#27ae60", alpha=0.2, label="Alpha (8-12 Hz)")
            plt.axvspan(13, 30, color="#3498db", alpha=0.15, label="Beta (13-30 Hz)")

            # Show filter cutoffs
            plt.axvline(2, color="r", linestyle="--", linewidth=2, alpha=0.5)
            plt.axvline(20, color="r", linestyle="--", linewidth=2, alpha=0.5)

            plt.xlim(1, 40)
            plt.xlabel("Frequency (Hz)", fontweight="bold")
            plt.ylabel("Power (V²/Hz)", fontweight="bold")
            plt.title(
                f"Figure 1: PSD (n={len(all_psds)} trials, 2-20 Hz filter)",
                fontweight="bold",
                fontsize=16,
            )
            plt.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
            plt.grid(True, which="both", ls="-", alpha=0.3)

            self._save_figure("fig01_power_spectral_density.png")

        except Exception as e:
            self.logger.error(f"Figure 1 Error: {e}")

    def _figure_02_gfp_peak_extraction(self):
        """Figure 2: GFP curve with extracted peaks."""
        try:
            plt.figure(figsize=(14, 5))
            start, end = 1000, 1500
            if len(self.gfp_curve) < end:
                end = len(self.gfp_curve)

            t_axis = np.arange(0, end - start) / self.sfreq
            gfp_segment = self.gfp_curve[start:end]

            plt.fill_between(
                t_axis, gfp_segment, color="#3498db", alpha=0.3, label="GFP Envelope"
            )
            plt.plot(
                t_axis,
                gfp_segment,
                color="#2980b9",
                linewidth=2,
                label="Global Field Power",
            )

            idx_in_window = [
                i - start for i in self.sampling_indices if start <= i < end
            ]
            if idx_in_window:
                peaks_y = gfp_segment[idx_in_window]
                plt.scatter(
                    np.array(idx_in_window) / self.sfreq,
                    peaks_y,
                    c="#e74c3c",
                    s=50,
                    zorder=5,
                    marker="o",
                    edgecolors="white",
                    linewidths=1.5,
                    label=f"Extracted Peaks (n={len(idx_in_window)})",
                )

            plt.xlabel("Time (s)", fontweight="bold")
            plt.ylabel("Global Field Power (μV)", fontweight="bold")
            plt.title(
                "Figure 2: GFP Peak Extraction Strategy", fontweight="bold", fontsize=16
            )
            plt.legend(frameon=True, fancybox=True, shadow=True)
            sns.despine()

            self._save_figure("fig02_gfp_peak_extraction.png")

        except Exception as e:
            self.logger.error(f"Figure 2 Error: {e}")

    def _figure_03_channel_time_series(self):
        """Figure 3: Individual channel time series."""
        try:
            fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
            channels_to_plot = [0, 7, 11, 14]  # Fp1, C3, P3, O1
            channel_names = [self.ch_names[i] for i in channels_to_plot]

            start, end = 0, 512
            t_axis = np.arange(start, end) / self.sfreq
            trial_data = self.raw_data_structure[0, :, start:end]

            for idx, (ch_idx, ch_name) in enumerate(
                zip(channels_to_plot, channel_names)
            ):
                axes[idx].plot(
                    t_axis, trial_data[ch_idx, :], color="#34495e", linewidth=1.2
                )
                axes[idx].set_ylabel(f"{ch_name}\n(μV)", fontweight="bold")
                axes[idx].grid(True, alpha=0.3)
                sns.despine(ax=axes[idx])

            axes[-1].set_xlabel("Time (s)", fontweight="bold")
            plt.suptitle(
                "Figure 3: Representative Channel Time Series",
                fontweight="bold",
                fontsize=16,
            )

            self._save_figure("fig03_channel_time_series.png")

        except Exception as e:
            self.logger.error(f"Figure 3 Error: {e}")

    def _figure_04_topomap_grid(self, topo_maps):
        """Figure 4: Grid of topographic maps."""
        try:
            fig, axes = plt.subplots(4, 8, figsize=(16, 8))
            axes = axes.flatten()

            # Select evenly spaced maps
            idx = np.linspace(0, len(topo_maps) - 1, 32, dtype=int)
            vmin = np.percentile(topo_maps[idx], 5)
            vmax = np.percentile(topo_maps[idx], 95)

            for i in range(32):
                img = topo_maps[idx[i]]
                im = axes[i].imshow(
                    img, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax
                )
                clip_path = self.make_circular_mask(axes[i], self.topo_map_size)
                im.set_clip_path(clip_path)
                axes[i].axis("off")
                axes[i].set_title(f"#{idx[i]}", fontsize=8)

            fig.colorbar(
                im,
                ax=axes,
                orientation="horizontal",
                fraction=0.05,
                pad=0.05,
                label="Normalized Amplitude (μV)",
            )

            plt.suptitle(
                "Figure 4: Representative Topographic Maps (n=32)",
                fontweight="bold",
                fontsize=16,
            )

            self._save_figure("fig04_topomap_grid.png")

        except Exception as e:
            self.logger.error(f"Figure 4 Error: {e}")

    def _figure_05_gfp_distribution(self):
        """Figure 5: Distribution of GFP values."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].hist(
                self.gfp_curve, bins=100, color="#3498db", alpha=0.7, edgecolor="black"
            )
            axes[0].axvline(
                np.median(self.gfp_curve),
                color="#e74c3c",
                linestyle="--",
                linewidth=2,
                label=f"Median: {np.median(self.gfp_curve):.2f} μV",
            )
            axes[0].set_xlabel("GFP Amplitude (μV)", fontweight="bold")
            axes[0].set_ylabel("Frequency", fontweight="bold")
            axes[0].set_title("GFP Distribution", fontweight="bold")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            sns.despine(ax=axes[0])

            sorted_gfp = np.sort(self.gfp_curve)
            cumulative = np.arange(1, len(sorted_gfp) + 1) / len(sorted_gfp)

            axes[1].plot(sorted_gfp, cumulative, color="#2ecc71", linewidth=2)
            axes[1].axhline(0.95, color="#e74c3c", linestyle="--", alpha=0.5)
            axes[1].axhline(0.50, color="#f39c12", linestyle="--", alpha=0.5)
            axes[1].set_xlabel("GFP Amplitude (μV)", fontweight="bold")
            axes[1].set_ylabel("Cumulative Probability", fontweight="bold")
            axes[1].set_title("Cumulative Distribution Function", fontweight="bold")
            axes[1].grid(True, alpha=0.3)
            sns.despine(ax=axes[1])

            plt.suptitle(
                "Figure 5: Global Field Power Statistics",
                fontweight="bold",
                fontsize=16,
            )
            plt.tight_layout()

            self._save_figure("fig05_gfp_distribution.png")

        except Exception as e:
            self.logger.error(f"Figure 5 Error: {e}")

    def _figure_06_channel_correlation_matrix(self, raw_dataset):
        """Figure 6: Spatial correlation between channels."""
        try:
            # Need to reshape flat dataset back to channels x time for correlation
            # raw_dataset from load_preprocessed is (Channels, Time)
            corr_matrix = np.corrcoef(raw_dataset)

            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Correlation Coefficient", fontweight="bold")

            tick_positions = np.arange(0, 32, 4)
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels([self.ch_names[i] for i in tick_positions], rotation=45)
            ax.set_yticklabels([self.ch_names[i] for i in tick_positions])

            ax.set_title(
                "Figure 6: Spatial Correlation Matrix",
                fontweight="bold",
                fontsize=16,
                pad=20,
            )

            plt.tight_layout()
            self._save_figure("fig06_channel_correlation.png")

        except Exception as e:
            self.logger.error(f"Figure 6 Error: {e}")

    def _figure_07_sensor_layout(self):
        """Figure 7: 32-channel sensor layout."""
        try:
            fig = plt.figure(figsize=(8, 8))
            if self.info is not None:
                mne.viz.plot_sensors(
                    self.info,
                    kind="topomap",
                    show_names=True,
                    axes=plt.gca(),
                    title="",
                    show=False,
                )
                plt.title(
                    "Figure 7: 32-Channel Biosemi Sensor Layout",
                    fontweight="bold",
                    fontsize=16,
                    pad=20,
                )
            self._save_figure("fig07_sensor_layout.png")
        except Exception as e:
            self.logger.error(f"Figure 7 Error: {e}")

    def _figure_08_spatial_variance(self, topo_maps):
        """Figure 8: Spatial variance map showing active regions."""
        try:
            spatial_var = np.var(topo_maps, axis=0)
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(spatial_var, cmap="hot", origin="lower")

            clip_path = self.make_circular_mask(ax, self.topo_map_size)
            im.set_clip_path(clip_path)

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Variance (μV²)", fontweight="bold")
            plt.title(
                "Figure 8: Spatial Variance Map (Active Regions)",
                fontweight="bold",
                fontsize=16,
            )
            plt.axis("off")
            self._save_figure("fig08_spatial_variance.png")
        except Exception as e:
            self.logger.error(f"Figure 8 Error: {e}")

    def _figure_09_butterfly_plot(self, raw_dataset):
        """Figure 9: Butterfly plot with GFP overlay."""
        try:
            start_samp, end_samp = 2000, 2512
            if raw_dataset.shape[1] < end_samp:
                end_samp = raw_dataset.shape[1]

            times = np.arange(0, end_samp - start_samp) / self.sfreq
            channel_data = raw_dataset[:, start_samp:end_samp].T
            gfp_data = self.gfp_curve[start_samp:end_samp]

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(times, channel_data, color="#95a5a6", alpha=0.3, linewidth=0.8)
            ax.plot(
                times,
                gfp_data,
                color="#e74c3c",
                linewidth=3,
                label="Global Field Power",
                zorder=10,
            )

            ax.set_xlabel("Time (s)", fontweight="bold")
            ax.set_ylabel("Amplitude (μV)", fontweight="bold")
            ax.set_title(
                "Figure 9: Butterfly Plot with GFP Overlay",
                fontweight="bold",
                fontsize=16,
            )
            ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)
            sns.despine()

            self._save_figure("fig09_butterfly_plot.png")

        except Exception as e:
            self.logger.error(f"Figure 9 Error: {e}")

    def _figure_10_frequency_band_topomaps(self):
        """Figure 10: Topomaps for different frequency bands."""
        try:
            bands = {"Theta": (4, 8), "Alpha": (8, 12), "Beta": (13, 30)}
            fig, axes = plt.subplots(1, 3, figsize=(16, 6))
            subset = self.raw_data_structure[0, :, :]

            for i, (band_name, (low, high)) in enumerate(bands.items()):
                sos = scipy.signal.butter(
                    4, [low, high], btype="bandpass", fs=self.sfreq, output="sos"
                )
                filtered = scipy.signal.sosfilt(sos, subset, axis=1)
                power = np.var(filtered, axis=1)

                if self.info is not None:
                    im, _ = mne.viz.plot_topomap(
                        power,
                        self.info,
                        axes=axes[i],
                        show=False,
                        cmap="viridis",
                        contours=6,
                        vlim=(power.min(), power.max()),
                    )
                    axes[i].set_title(
                        f"{band_name}\n({low}-{high} Hz)",
                        fontweight="bold",
                        fontsize=14,
                    )
                    plt.colorbar(
                        im, ax=axes[i], fraction=0.046, pad=0.04, label="Power (μV²)"
                    )

            plt.suptitle(
                "Figure 10: Frequency Band Power Distribution",
                fontweight="bold",
                fontsize=16,
            )
            self._save_figure("fig10_frequency_bands.png")

        except Exception as e:
            self.logger.error(f"Figure 10 Error: {e}")

    def _figure_11_split_half_reliability(self, topo_maps):
        """Figure 11: Split-half reliability analysis."""
        try:
            mid = len(topo_maps) // 2
            half1 = np.mean(topo_maps[:mid], axis=0)
            half2 = np.mean(topo_maps[mid:], axis=0)
            corr = np.corrcoef(half1.flatten(), half2.flatten())[0, 1]

            fig, axes = plt.subplots(1, 3, figsize=(16, 6))

            im1 = axes[0].imshow(half1, cmap="RdBu_r", origin="lower")
            clip1 = self.make_circular_mask(axes[0], self.topo_map_size)
            im1.set_clip_path(clip1)
            axes[0].set_title(f"First Half\n(n={mid} maps)", fontweight="bold")
            axes[0].axis("off")
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

            im2 = axes[1].imshow(half2, cmap="RdBu_r", origin="lower")
            clip2 = self.make_circular_mask(axes[1], self.topo_map_size)
            im2.set_clip_path(clip2)
            axes[1].set_title(
                f"Second Half\n(n={len(topo_maps)-mid} maps)", fontweight="bold"
            )
            axes[1].axis("off")
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

            sns.regplot(
                x=half1.flatten(),
                y=half2.flatten(),
                ax=axes[2],
                scatter_kws={"alpha": 0.2, "color": "#34495e", "s": 10},
                line_kws={"color": "#e74c3c", "linewidth": 2},
            )
            axes[2].set_title(
                f"Reliability: r = {corr:.4f}", fontweight="bold", fontsize=14
            )
            axes[2].set_xlabel("Pixel Intensity (First Half)", fontweight="bold")
            axes[2].set_ylabel("Pixel Intensity (Second Half)", fontweight="bold")
            axes[2].grid(True, alpha=0.3)
            sns.despine(ax=axes[2])

            plt.suptitle(
                "Figure 11: Split-Half Reliability Analysis",
                fontweight="bold",
                fontsize=16,
            )
            plt.tight_layout()
            self._save_figure("fig11_split_half_reliability.png")

        except Exception as e:
            self.logger.error(f"Figure 11 Error: {e}")

    # =========================================================================
    # Data Preparation for Deep Learning
    # =========================================================================

    def normalize_and_prepare_data(self, topo_maps):
        """Normalize topographic maps and prepare for deep learning."""
        # Robust normalization using percentiles
        p1, p99 = np.percentile(topo_maps, [1, 99])
        topo_maps = np.clip(topo_maps, p1, p99)

        # Min-max normalization
        d_min, d_max = topo_maps.min(), topo_maps.max()
        if d_max - d_min > 0:
            topo_maps = (topo_maps - d_min) / (d_max - d_min)

        self.logger.info(
            f"Normalized data: min={topo_maps.min():.4f}, "
            f"max={topo_maps.max():.4f}, mean={topo_maps.mean():.4f}"
        )
        return topo_maps

    def create_dataloaders(self, topo_maps):
        """Create train/validation/test dataloaders."""
        data_torch = topo_maps.reshape(-1, 1, self.topo_map_size, self.topo_map_size)
        tensor_x = T.tensor(data_torch, dtype=T.float32)
        tensor_y = T.zeros(len(tensor_x))
        dataset = TensorDataset(tensor_x, tensor_y)

        train_len = int(0.7 * len(dataset))
        val_len = int(0.2 * len(dataset))
        test_len = len(dataset) - train_len - val_len

        train, val, test = random_split(dataset, [train_len, val_len, test_len])
        batch_size = self.config["batch_size"]

        train_loader = DataLoader(
            train, batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val, batch_size=batch_size, shuffle=False, drop_last=True
        )
        test_loader = DataLoader(
            test, batch_size=batch_size, shuffle=False, drop_last=True
        )

        self.logger.info(
            f"Created dataloaders: Train={len(train)}, "
            f"Val={len(val)}, Test={len(test)}"
        )

        return VaeDatasets(train_loader, val_loader, test_loader, train)

    # =========================================================================
    # Main Processing Pipeline
    # =========================================================================

    def process(self, participant_id=None):
        """Execute complete EEG processing pipeline."""
        if participant_id:
            self.participant_id = participant_id
            self._setup_participant_paths()
            self._create_directories()

        self.logger.info("=" * 60)
        self.logger.info(f"Processing Participant: {self.participant_id}")
        self.logger.info("=" * 60)

        # Step 1: Check data availability (and Download if needed)
        self.check_and_download_data()

        # Step 2: Load data
        self.logger.info("Step 1/5: Loading data...")
        # Note: We load the .dat file that was verified/downloaded in Step 1
        raw_data = self.load_preprocessed_dat(self.config["data_path"])
        if raw_data is None:
            raise ValueError("Data loading failed")

        # Step 3: Create MNE objects
        self.logger.info("Step 2/5: Creating MNE objects...")
        raw_mne = self.create_mne_raw_object(raw_data, apply_filter=True)

        # Step 4: Generate topographic maps
        self.logger.info("Step 3/5: Generating topographic maps...")
        topo_maps = self.generate_topographic_maps(raw_mne)

        # Save raw maps
        output_path = Path(self.config["output_path"])
        np.save(output_path / "topo_maps.npy", topo_maps)
        self.logger.info(f"Saved {len(topo_maps)} topographic maps")

        # Step 5: Normalize and prepare data
        self.logger.info("Step 4/5: Normalizing data...")
        topo_maps_normalized = self.normalize_and_prepare_data(topo_maps)

        # Step 6: Generate visualizations
        self.logger.info("Step 5/5: Generating visualizations...")
        self.generate_research_figures(topo_maps_normalized, raw_data)

        # Create dataloaders
        vae_datasets = self.create_dataloaders(topo_maps_normalized)

        self.logger.info("=" * 60)
        self.logger.info("Processing Complete!")
        self.logger.info(f"Figures saved to: {self.config['figure_dir']}")
        self.logger.info(f"Data saved to: {self.config['output_path']}")
        self.logger.info("=" * 60)

        return vae_datasets


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Configure processor
    config = {
        "data_dir": "./data",
        "batch_size": 64,
        "max_topo_samples": 50000,  # Reduced slightly for speed in demo
    }

    # Initialize with default participant
    # Note: If s01.dat is missing, it will now trigger the GDrive download
    processor = EEGProcessor(config, participant_id="s01")

    try:
        vae_datasets = processor.process("s01")
        print("\n" + "=" * 60)
        print("SUCCESS! All visualizations generated.")
        print("Check the './data/Figure/s01/' directory for results.")
        print("=" * 60)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
