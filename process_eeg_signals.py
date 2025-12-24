#!/usr/bin/env python3
"""
Advanced Research EEG Processor for Microstate Analysis
-------------------------------------------------------
Complete implementation with 12 publication-ready visualizations
Backend: Agg (Non-interactive, saves files only)
Visualization: MNE Standard (Head outlines, sensors)
"""

import sys
import os
import pickle
import logging
import warnings
import math as m
from datetime import datetime
from pathlib import Path

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
    handlers=[logging.StreamHandler(sys.stdout)],  # Force output to stdout
)
logge = logging.getLogger("eeg_processor")

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

        Parameters
        ----------
        config_dict : dict, optional
            Configuration parameters
        logger : logging.Logger, optional
            Custom logger instance
        participant_id : str, optional
            Participant identifier
        """
        self.logger = logger or logging.getLogger("eeg_processor")

        # Default configuration
        self.config = {
            "data_dir": "./data",
            "output_path": "./Topomaps",
            "figure_dir": "./Figure",
            "sample_freq": 128,
            "num_eeg_channels": 32,
            "topo_map_size": 40,
            "interpolation_method": "cubic",
            "batch_size": 64,
            "max_topo_samples": 500000,
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
        for key in ["figure_dir", "output_path"]:
            Path(self.config.get(key)).mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Data Loading
    # =========================================================================
    def check_and_download_data(self):
        """Verify that data file exists."""
        data_dir = self.config.get("data_dir")
        Path(data_dir).mkdir(parents=True, exist_ok=True)

        participant_file = (
            f"{data_dir}/{self.participant_id}.dat" if self.participant_id else None
        )

        if participant_file and os.path.exists(participant_file):
            self.logger.info(f"Data found: {participant_file}")
            return

        self.logger.warning(f"Data NOT found at {participant_file}")

    def load_preprocessed_dat(self, dat_file_path):
        """
        Load ALREADY-PREPROCESSED .dat file with proper trial handling.

        The data has already been:
        - Filtered (4-45 Hz)
        - Average referenced
        - EOG removed
        - Segmented into trials

        We add padding between trials to avoid edge effects.
        """
        try:
            with open(dat_file_path, "rb") as f:
                data_dict = pickle.load(f, encoding="latin1")

            data = data_dict["data"]
            eeg_data = data[:, :32, :]
            n_trials, n_channels, n_timepoints = eeg_data.shape

            self.raw_data_structure = eeg_data

            # Add padding between trials to avoid edge artifacts
            padding_samples = int(
                self.config.get("trial_padding_sec", 0.5) * self.sfreq
            )

            padded_trials = []
            for trial_idx in range(n_trials):
                trial = eeg_data[trial_idx]  # (channels, timepoints)
                padded_trials.append(trial)

                if trial_idx < n_trials - 1:
                    padding = np.zeros((n_channels, padding_samples))
                    padded_trials.append(padding)

            reshaped_data = np.hstack(padded_trials)

            self.logger.info(
                f"Loaded PREPROCESSED data: {n_trials} trials, {n_channels} channels, "
                f"{n_timepoints} timepoints/trial"
            )
            self.logger.info(f"Added {padding_samples} samples padding between trials")
            self.logger.info(
                "Data preprocessing status:\n"
                "  ✓ Downsampled to 128Hz\n"
                "  ✓ EOG artifacts removed\n"
                "  ✓ Bandpass filtered 4-45 Hz\n"
                "  ✓ Average referenced\n"
                "  ✓ Segmented into trials"
            )

            return reshaped_data

        except Exception as e:
            self.logger.error(f"Error loading .dat file: {e}")
            return None

    # =========================================================================
    # MNE Object Creation
    # =========================================================================

    def create_mne_raw_object(self, data, apply_filter=True):
        """
        Create MNE Raw object with montage and preprocessing.

        Parameters
        ----------
        data : np.ndarray
            EEG data (channels x samples)
        apply_filter : bool
            Whether to apply spatial filter

        Returns
        -------
        mne.io.RawArray
            MNE Raw object
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
        raw.filter(
            l_freq=2.0, h_freq=20.0, method="fir", fir_design="firwin", phase="zero"
        )
        self.logger.info("Applied 2-20 Hz bandpass filter")

        # Apply average reference (critical for microstate analysis)
        # Commented out because its already done in the preprocessing
        # raw.set_eeg_reference('average', projection=False)
        # self.logger.info("Applied average reference")

        # Apply spatial filter to reduce local artifacts
        if apply_filter:
            apply_spatial_filter(raw, n_jobs=-1)
            self.logger.info("Applied spatial filter")

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
        """
        Extract 3D electrode coordinates from montage.

        Parameters
        ----------
        montage_channel_location : list
            Digitization points from MNE info

        Returns
        -------
        np.ndarray
            3D coordinates of shape (n_channels, 3)
        """
        location = []
        locs = montage_channel_location[-32:]

        for i in range(32):
            vals = list(locs[i].values())
            location.append(vals[1] * 1000)

        return np.array(location)

    # =========================================================================
    # Topographic Map Generation
    # =========================================================================

    def create_topographic_map(self, channel_values, pos_2d):
        """
        Create interpolated topographic map.

        Parameters
        ----------
        channel_values : np.ndarray
            Electrode values
        pos_2d : np.ndarray
            2D electrode positions

        Returns
        -------
        np.ndarray
            Interpolated topographic map
        """
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
        """
        Generate topographic maps from GFP peaks.

        Parameters
        ----------
        raw_mne : mne.io.RawArray
            MNE Raw object

        Returns
        -------
        np.ndarray
            Array of topographic maps
        """
        # Get electrode positions
        self.pos_3d = self.get_3d_coordinates(raw_mne.info["dig"])
        self.pos_2d = np.array([self.azim_proj(p) for p in self.pos_3d])

        # Extract GFP peaks
        self.logger.info("Extracting GFP peaks...")
        print("Extracting GFP peaks...")
        gfp_peaks_raw = extract_gfp_peaks(raw_mne, min_peak_distance=3)

        # Get peak indices
        peak_samples = gfp_peaks_raw.get_data().flatten().astype(int)
        self.logger.info(f"Found {len(peak_samples)} GFP peaks")
        print("Stopped Extracting GFP peaks...")

        max_samples = self.config.get("max_topo_samples", 50000)
        if len(peak_samples) > max_samples:
            selected = np.random.choice(len(peak_samples), max_samples, replace=False)
            selected.sort()
            selected_samples = peak_samples[selected]
        else:
            selected_samples = peak_samples

        self.logger.info(
            f"Generating {len(selected_samples)} maps from "
            f"{len(peak_samples)} GFP peaks (standard approach)"
        )

        # Get data
        data = raw_mne.get_data()

        # Generate maps
        maps = []
        for sample_idx in selected_samples:
            vals = data[:, sample_idx]
            img = self.create_topographic_map(vals, self.pos_2d)
            maps.append(img)

        # Calculate GFP
        gfp = np.std(data, axis=0)

        self.sampling_indices = selected_samples
        self.gfp_curve = gfp

        return np.array(maps)

    # =========================================================================
    # Visualization Helpers
    # =========================================================================

    def make_circular_mask(self, ax, img_size):
        """
        Create circular mask and head outline on axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to add mask to
        img_size : int
            Size of image

        Returns
        -------
        matplotlib.patches.Circle
            Circle patch for clipping
        """
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
        """
        Generate all 12 publication-ready figures.

        Parameters
        ----------
        topo_maps : np.ndarray
            Generated topographic maps
        raw_dataset : np.ndarray
            Raw EEG data
        """
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
        """FIXED: Figure 1 using ALL trials for average PSD."""
        try:
            plt.figure(figsize=(10, 6))

            # FIXED: Calculate PSD for ALL trials and average
            all_psds = []
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

            # Select time window
            start, end = 1000, 1500
            t_axis = np.arange(0, end - start) / self.sfreq
            gfp_segment = self.gfp_curve[start:end]

            # Plot GFP
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

            # Mark extracted peaks
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

            # Select channels to display (frontal, central, parietal, occipital)
            channels_to_plot = [0, 7, 11, 14]  # Fp1, C3, P3, O1
            channel_names = [self.ch_names[i] for i in channels_to_plot]

            # Time window
            start, end = 0, 512
            t_axis = np.arange(start, end) / self.sfreq

            # Get data
            trial_data = self.raw_data_structure[0, :, start:end]

            # Plot each channel
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

            # Find global min/max for consistent colorbar
            vmin = np.percentile(topo_maps[idx], 5)
            vmax = np.percentile(topo_maps[idx], 95)

            for i in range(32):
                img = topo_maps[idx[i]]
                im = axes[i].imshow(
                    img, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax
                )

                # Apply circular mask
                clip_path = self.make_circular_mask(axes[i], self.topo_map_size)
                im.set_clip_path(clip_path)

                axes[i].axis("off")
                axes[i].set_title(f"#{idx[i]}", fontsize=8)

            # Add colorbar
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

            # Histogram
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

            # Cumulative distribution
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
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(raw_dataset)

            fig, ax = plt.subplots(figsize=(12, 10))

            # Plot correlation matrix
            im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Correlation Coefficient", fontweight="bold")

            # Set ticks
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
            # Calculate variance across all maps
            spatial_var = np.var(topo_maps, axis=0)

            fig, ax = plt.subplots(figsize=(8, 7))

            im = ax.imshow(spatial_var, cmap="hot", origin="lower")

            # Apply circular mask
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
            # Select time window
            start_samp, end_samp = 2000, 2512
            times = np.arange(0, end_samp - start_samp) / self.sfreq
            channel_data = raw_dataset[:, start_samp:end_samp].T
            gfp_data = self.gfp_curve[start_samp:end_samp]

            fig, ax = plt.subplots(figsize=(14, 6))

            # Plot all channels
            ax.plot(times, channel_data, color="#95a5a6", alpha=0.3, linewidth=0.8)

            # Overlay GFP
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
                # Bandpass filter
                sos = scipy.signal.butter(
                    4, [low, high], btype="bandpass", fs=self.sfreq, output="sos"
                )
                filtered = scipy.signal.sosfilt(sos, subset, axis=1)

                # Calculate power
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
            # Split data
            mid = len(topo_maps) // 2
            half1 = np.mean(topo_maps[:mid], axis=0)
            half2 = np.mean(topo_maps[mid:], axis=0)

            # Calculate correlation
            corr = np.corrcoef(half1.flatten(), half2.flatten())[0, 1]

            fig, axes = plt.subplots(1, 3, figsize=(16, 6))

            # First half
            im1 = axes[0].imshow(half1, cmap="RdBu_r", origin="lower")
            clip1 = self.make_circular_mask(axes[0], self.topo_map_size)
            im1.set_clip_path(clip1)
            axes[0].set_title(f"First Half\n(n={mid} maps)", fontweight="bold")
            axes[0].axis("off")
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

            # Second half
            im2 = axes[1].imshow(half2, cmap="RdBu_r", origin="lower")
            clip2 = self.make_circular_mask(axes[1], self.topo_map_size)
            im2.set_clip_path(clip2)
            axes[1].set_title(
                f"Second Half\n(n={len(topo_maps)-mid} maps)", fontweight="bold"
            )
            axes[1].axis("off")
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

            # Correlation plot
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

    def _figure_12_spatial_filter_comparison(self, raw_dataset):
        """Figure 12: Effect of spatial filtering."""
        try:
            if self.raw_mne_unfiltered is None:
                self.logger.warning("Unfiltered data not available for comparison")
                return

            # Select random timepoints
            n_samples = 8
            random_samples = sorted(
                np.random.randint(0, raw_dataset.shape[1], n_samples)
            )

            fig, axes = plt.subplots(2, n_samples, figsize=(18, 6))

            # Get data from both versions
            unfiltered_data = self.raw_mne_unfiltered.get_data()
            filtered_data = raw_dataset

            # Find global min/max for consistent colorbar
            vmin = min(
                np.percentile(unfiltered_data[:, random_samples], 5),
                np.percentile(filtered_data[:, random_samples], 5),
            )
            vmax = max(
                np.percentile(unfiltered_data[:, random_samples], 95),
                np.percentile(filtered_data[:, random_samples], 95),
            )

            for s, sample in enumerate(random_samples):
                # Unfiltered
                mne.viz.plot_topomap(
                    unfiltered_data[:, sample],
                    pos=self.raw_mne_unfiltered.info,
                    axes=axes[0, s],
                    sphere=np.array([0, 0, 0, 0.1]),
                    show=False,
                    cmap="RdBu_r",
                    vlim=(vmin, vmax),
                )
                axes[0, s].set_title(f"t={sample}", fontsize=10)

                # Filtered
                mne.viz.plot_topomap(
                    filtered_data[:, sample],
                    pos=self.info,
                    axes=axes[1, s],
                    sphere=np.array([0, 0, 0, 0.1]),
                    show=False,
                    cmap="RdBu_r",
                    vlim=(vmin, vmax),
                )

            axes[0, 0].set_ylabel(
                "Without Spatial Filter", fontweight="bold", fontsize=12
            )
            axes[1, 0].set_ylabel("With Spatial Filter", fontweight="bold", fontsize=12)

            plt.suptitle(
                "Figure 12: Spatial Filter Effect on Topographic Maps",
                fontweight="bold",
                fontsize=16,
            )
            fig.tight_layout()

            self._save_figure("fig12_spatial_filter_comparison.png")

        except Exception as e:
            self.logger.error(f"Figure 12 Error: {e}")

    # =========================================================================
    # Data Preparation for Deep Learning
    # =========================================================================

    def normalize_and_prepare_data(self, topo_maps):
        """
        Normalize topographic maps and prepare for deep learning.

        Parameters
        ----------
        topo_maps : np.ndarray
            Raw topographic maps

        Returns
        -------
        np.ndarray
            Normalized maps
        """
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
        """
        Create train/validation/test dataloaders.

        Parameters
        ----------
        topo_maps : np.ndarray
            Normalized topographic maps

        Returns
        -------
        VaeDatasets
            Container with dataloaders
        """
        # Reshape for PyTorch (N, C, H, W)
        data_torch = topo_maps.reshape(-1, 1, self.topo_map_size, self.topo_map_size)

        tensor_x = T.tensor(data_torch, dtype=T.float32)
        tensor_y = T.zeros(len(tensor_x))

        dataset = TensorDataset(tensor_x, tensor_y)

        # Split dataset (70% train, 20% val, 10% test)
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
        """
        Execute complete EEG processing pipeline.

        Parameters
        ----------
        participant_id : str, optional
            Participant identifier

        Returns
        -------
        VaeDatasets
            Dataloaders for training
        """
        if participant_id:
            self.participant_id = participant_id
            self._setup_participant_paths()
            self._create_directories()

        self.logger.info("=" * 60)
        self.logger.info(f"Processing Participant: {self.participant_id}")
        self.logger.info("=" * 60)

        # Step 1: Check data availability
        print("Checking availability")
        self.check_and_download_data()

        # Step 2: Load data
        self.logger.info("Step 1/5: Loading data...")
        print("Step 1/5: Loading data...")
        raw_data = self.load_preprocessed_dat(self.config["data_path"])
        if raw_data is None:
            raise ValueError("Data loading failed")

        # Step 3: Create MNE objects
        self.logger.info("Step 2/5: Creating MNE objects...")
        print("Step 2/5: Creating MNE objects...")

        # Create unfiltered version for comparison
        # self.raw_mne_unfiltered = self.create_mne_raw_object(
        #     raw_data.copy(),
        #     apply_filter=False
        # )

        # Create filtered version for analysis
        raw_mne = self.create_mne_raw_object(raw_data, apply_filter=True)

        # Step 4: Generate topographic maps
        self.logger.info("Step 3/5: Generating topographic maps...")
        print("Step 3/5: Generating topographic maps...")
        topo_maps = self.generate_topographic_maps(raw_mne)

        # Save raw maps
        output_path = Path(self.config["output_path"])
        np.save(output_path / "topo_maps.npy", topo_maps)
        self.logger.info(f"Saved {len(topo_maps)} topographic maps")
        print(f"Saved {len(topo_maps)} topographic maps")

        # Step 5: Normalize and prepare data
        self.logger.info("Step 4/5: Normalizing data...")
        print("Step 4/5: Normalizing data...")
        topo_maps_normalized = self.normalize_and_prepare_data(topo_maps)

        # Step 6: Generate visualizations
        self.logger.info("Step 5/5: Generating visualizations...")
        print("Step 5/5: Generating visualizations...")
        self.generate_research_figures(topo_maps_normalized, raw_data)
        print("Figures generated")

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
        "max_topo_samples": 100,
    }

    # Initialize and run
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
