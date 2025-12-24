import os
import pickle
import logging
import warnings
import math as m
from datetime import datetime

# --- CRITICAL: Set Backend to Non-Interactive ---
import matplotlib

matplotlib.use("Agg")  # Prevents pop-up windows
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

# Deep Learning
import torch as T
from torch.utils.data import DataLoader, TensorDataset, random_split

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("eeg_processor")

warnings.filterwarnings("ignore")
logging.getLogger("mne").setLevel(logging.WARNING)

# --- VISUALIZATION STYLE SETTINGS ---
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
    def __init__(self, train, validation, test, train_set=None):
        self.train = train
        self.validation = validation
        self.test = test
        self.train_set = train_set


class EEGProcessor:
    def __init__(self, config_dict=None, logger=None, participant_id=None):
        self.logger = logger or logging.getLogger("eeg_processor")

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
        if participant_id:
            data_dir = self.config.get("data_dir", "./data")
            self.config["data_path"] = f"{data_dir}/{participant_id}.dat"
            self.config["output_path"] = f"{data_dir}/Topomaps/{participant_id}"
            self.config["figure_dir"] = f"{data_dir}/Figure/{participant_id}"

        self.sfreq = self.config.get("sample_freq", 128)
        self.topo_map_size = self.config.get("topo_map_size", 40)

        os.makedirs(self.config.get("figure_dir"), exist_ok=True)
        os.makedirs(self.config.get("output_path"), exist_ok=True)

        self.ch_names = [
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

        self.info = None

    def check_and_download_data(self):
        data_dir = self.config.get("data_dir")
        os.makedirs(data_dir, exist_ok=True)
        participant_file = (
            f"{data_dir}/{self.participant_id}.dat" if self.participant_id else None
        )
        if participant_file and os.path.exists(participant_file):
            self.logger.info(f"Data found: {participant_file}")
            return
        self.logger.warning(f"Data NOT found at {participant_file}")

    def load_preprocessed_dat(self, dat_file_path):
        try:
            with open(dat_file_path, "rb") as f:
                data_dict = pickle.load(f, encoding="latin1")
            data = data_dict["data"]
            eeg_data = data[:, :32, :]
            n_trials, n_channels, n_timepoints = eeg_data.shape
            reshaped_data = eeg_data.reshape(n_trials * n_timepoints, n_channels).T
            self.raw_data_structure = eeg_data
            return reshaped_data
        except Exception as e:
            self.logger.error(f"Error loading .dat file: {e}")
            return None

    def create_mne_raw_object(self, data):
        if data.shape[0] > data.shape[1]:
            data = data.T
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)
        montage = mne.channels.make_standard_montage("biosemi32")
        raw.set_montage(montage)
        self.info = raw.info
        return raw

    # --- Geometry & Interpolation ---
    def azim_proj(self, pos):
        [r, elev, az] = self.cart2sph(pos[0], pos[1], pos[2])
        return self.pol2cart(az, m.pi / 2 - elev)

    def cart2sph(self, x, y, z):
        x2_y2 = x**2 + y**2
        r = m.sqrt(x2_y2 + z**2)
        elev = m.atan2(z, m.sqrt(x2_y2))
        az = m.atan2(y, x)
        return r, elev, az

    def pol2cart(self, theta, rho):
        return rho * m.cos(theta), rho * m.sin(theta)

    def get_3d_coordinates(self, montage_channel_location):
        location = []
        locs = montage_channel_location[-32:]
        for i in range(32):
            vals = list(locs[i].values())
            location.append(vals[1] * 1000)
        return np.array(location)

    def create_topographic_map(self, channel_values, pos_2d):
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

    def generate_topographic_maps(self, raw_dataset, raw_mne):
        self.pos_3d = self.get_3d_coordinates(raw_mne.info["dig"])
        self.pos_2d = np.array([self.azim_proj(p) for p in self.pos_3d])

        gfp = np.std(raw_dataset, axis=0)
        peaks, _ = scipy.signal.find_peaks(gfp, distance=3)

        max_samples = self.config.get("max_topo_samples", 50000)
        if len(peaks) > max_samples:
            indices = np.random.choice(peaks, max_samples, replace=False)
            indices.sort()
        else:
            indices = peaks

        self.logger.info(f"Generating {len(indices)} maps...")
        maps = []
        for i, idx in enumerate(indices):
            vals = raw_dataset[:, idx]
            img = self.create_topographic_map(vals, self.pos_2d)
            maps.append(img)

        self.sampling_indices = indices
        self.gfp_curve = gfp
        return np.array(maps)

    # --- HELPER: Circular Mask for Square Arrays ---
    def make_circular_mask(self, ax, img_size):
        center = img_size / 2 - 0.5
        radius = img_size / 2
        circle = patches.Circle((center, center), radius, transform=ax.transData)
        head_circle = patches.Circle(
            (center, center),
            radius,
            linewidth=2,
            edgecolor="black",
            facecolor="none",
            zorder=10,
        )
        ax.add_patch(head_circle)
        nose_len = radius * 0.15
        nose_wid = radius * 0.1
        nose_x = [center - nose_wid, center, center + nose_wid]
        nose_y = [img_size - 1, img_size - 1 + nose_len, img_size - 1]
        ax.plot(nose_x, nose_y, color="black", linewidth=2, zorder=10)
        return circle

    # --- RESEARCH FIGURES (ALL 11) ---
    def generate_research_figures(self, topo_maps, raw_dataset):
        self.logger.info("Generating All 11 Research Visualizations (Silent Mode)...")
        fig_dir = self.config["figure_dir"]

        # --- 1. PSD ---
        try:
            plt.figure(figsize=(8, 5))
            trial_data = self.raw_data_structure[0, :, :]
            f, Pxx = scipy.signal.welch(trial_data, fs=self.sfreq, nperseg=512, axis=1)
            mean_psd = np.mean(Pxx, axis=0)
            plt.semilogy(f, mean_psd, color="#333333", linewidth=1.5)
            plt.axvspan(8, 12, color="#2ecc71", alpha=0.2, label="Alpha")
            plt.axvspan(4, 8, color="#f1c40f", alpha=0.1, label="Theta")
            plt.xlim(1, 40)
            plt.xlabel("Frequency (Hz)", fontweight="bold")
            plt.ylabel("PSD (V²/Hz)", fontweight="bold")
            plt.title("Fig 1: Global Power Spectral Density", fontweight="bold")
            plt.legend()
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(f"{fig_dir}/fig1_psd.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Fig 1 Error: {e}")

        # --- 2. GFP Sampling ---
        try:
            plt.figure(figsize=(12, 4))
            start, end = 1000, 1500
            t_axis = np.arange(0, end - start) / self.sfreq
            gfp_segment = self.gfp_curve[start:end]
            plt.fill_between(t_axis, gfp_segment, color="#3498db", alpha=0.3)
            plt.plot(t_axis, gfp_segment, color="#2980b9", linewidth=1.5)
            idx_in_window = [
                i - start for i in self.sampling_indices if start <= i < end
            ]
            peaks_y = gfp_segment[idx_in_window]
            plt.scatter(
                np.array(idx_in_window) / self.sfreq, peaks_y, c="red", s=30, zorder=5
            )
            plt.xlabel("Time (s)", fontweight="bold")
            plt.ylabel("GFP (μV)", fontweight="bold")
            plt.title("Fig 2: GFP Peak Extraction", fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{fig_dir}/fig2_gfp_sampling.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Fig 2 Error: {e}")

        # --- 3. PIXEL HISTOGRAM (Restored) ---
        try:
            plt.figure(figsize=(7, 5))
            flat = topo_maps.flatten()
            sample = np.random.choice(flat, 50000)
            sns.histplot(
                sample, bins=50, kde=True, color="#34495e", line_kws={"linewidth": 2}
            )
            plt.xlabel("Normalized Pixel Intensity", fontweight="bold")
            plt.ylabel("Count", fontweight="bold")
            plt.title("Fig 3: Topomap Input Distribution", fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{fig_dir}/fig3_distribution.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Fig 3 Error: {e}")

        # --- 4. Topomap Grid ---
        try:
            fig, ax = plt.subplots(4, 8, figsize=(14, 7))
            ax = ax.flatten()
            idx = np.linspace(0, len(topo_maps) - 1, 32, dtype=int)
            for i in range(32):
                im = ax[i].imshow(topo_maps[idx[i]], cmap="RdBu_r", origin="lower")
                clip_path = self.make_circular_mask(ax[i], self.topo_map_size)
                im.set_clip_path(clip_path)
                ax[i].axis("off")
            plt.suptitle(
                "Fig 4: Extracted Topomap Samples", fontweight="bold", fontsize=16
            )
            plt.savefig(f"{fig_dir}/fig4_grid.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Fig 4 Error: {e}")

        # --- 5. PCA SCREE PLOT (Restored) ---
        try:
            # Use data at GFP peaks for valid dimensionality reduction
            peak_data = raw_dataset[:, self.sampling_indices]
            # Standardize
            peak_data = (
                peak_data - np.mean(peak_data, axis=1, keepdims=True)
            ) / np.std(peak_data, axis=1, keepdims=True)
            U, s, Vh = scipy.linalg.svd(peak_data, full_matrices=False)
            explained_variance = (s**2) / (len(s) - 1)
            cumulative_variance = np.cumsum(
                explained_variance / explained_variance.sum()
            )

            plt.figure(figsize=(8, 5))
            plt.plot(
                np.arange(1, 21),
                cumulative_variance[:20] * 100,
                "o-",
                color="#8e44ad",
                linewidth=2,
            )
            plt.axhline(
                y=70, color="#e74c3c", linestyle="--", alpha=0.7, label="70% Variance"
            )
            plt.xlabel("Principal Component", fontweight="bold")
            plt.ylabel("Cumulative Explained Variance (%)", fontweight="bold")
            plt.title("Fig 5: PCA Scree Plot (Dimensionality Check)", fontweight="bold")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{fig_dir}/fig5_scree_plot.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Fig 5 Error: {e}")

        # --- 6. GMD vs GFP (Restored) ---
        try:
            seg_len = 500
            start_idx = 2000
            if raw_dataset.shape[1] > start_idx + seg_len:
                seg_data = raw_dataset[:, start_idx : start_idx + seg_len]
                # Normalized GMD calculation
                seg_norm = seg_data / (
                    np.linalg.norm(seg_data, axis=0, keepdims=True) + 1e-8
                )
                gmd = np.linalg.norm(seg_norm[:, 1:] - seg_norm[:, :-1], axis=0)
                gfp_seg = np.std(seg_data, axis=0)[1:]

                fig, ax1 = plt.subplots(figsize=(10, 5))

                # Plot GFP
                ax1.plot(gfp_seg, color="#e74c3c", linewidth=2, label="GFP (Power)")
                ax1.set_xlabel("Time (samples)", fontweight="bold")
                ax1.set_ylabel("GFP", color="#e74c3c", fontweight="bold")
                ax1.tick_params(axis="y", labelcolor="#e74c3c")

                # Plot GMD on twin axis
                ax2 = ax1.twinx()
                ax2.plot(
                    gmd,
                    color="#3498db",
                    linestyle="--",
                    linewidth=1.5,
                    label="GMD (Dissimilarity)",
                )
                ax2.set_ylabel("GMD", color="#3498db", fontweight="bold")
                ax2.tick_params(axis="y", labelcolor="#3498db")

                plt.title(
                    "Fig 6: GFP Peaks vs Map Dissimilarity (Microstate Stability)",
                    fontweight="bold",
                )
                fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
                plt.tight_layout()
                plt.savefig(f"{fig_dir}/fig6_gfp_vs_gmd.png")
                plt.close()
        except Exception as e:
            self.logger.error(f"Fig 6 Error: {e}")

        # --- 7. Sensor Layout ---
        try:
            fig = plt.figure(figsize=(6, 6))
            if self.info is not None:
                mne.viz.plot_sensors(
                    self.info,
                    kind="topomap",
                    show_names=True,
                    axes=plt.gca(),
                    title="Fig 7: Sensor Layout",
                    show=False,
                )
                plt.tight_layout()
                plt.savefig(f"{fig_dir}/fig7_sensor_layout.png")
                plt.close()
        except Exception as e:
            self.logger.error(f"Fig 7 Error: {e}")

        # --- 8. Spatial Activation ---
        try:
            spatial_var = np.var(topo_maps, axis=0)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(spatial_var, cmap="inferno", origin="lower")
            clip_path = self.make_circular_mask(ax, self.topo_map_size)
            im.set_clip_path(clip_path)
            plt.colorbar(im, label="Pixel Variance")
            plt.title("Fig 8: Spatial Variance", fontweight="bold")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"{fig_dir}/fig8_activation_map.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Fig 8 Error: {e}")

        # --- 9. Butterfly Plot ---
        try:
            start_samp, end_samp = 2000, 2512
            times = np.arange(0, end_samp - start_samp) / self.sfreq
            channel_data = raw_dataset[:, start_samp:end_samp].T
            gfp_data = self.gfp_curve[start_samp:end_samp]
            plt.figure(figsize=(12, 6))
            plt.plot(times, channel_data, color="k", alpha=0.15, linewidth=0.5)
            plt.plot(times, gfp_data, color="#e74c3c", linewidth=2, label="GFP")
            plt.xlabel("Time (s)", fontweight="bold")
            plt.ylabel("Amplitude (μV)", fontweight="bold")
            plt.title("Fig 9: Butterfly Plot", fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{fig_dir}/fig9_butterfly_plot.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Fig 9 Error: {e}")

        # --- 10. Frequency Bands ---
        try:
            bands = {"Theta": (4, 8), "Alpha": (8, 12), "Beta": (13, 30)}
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            subset = self.raw_data_structure[0, :, :]
            for i, (band, (low, high)) in enumerate(bands.items()):
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
                        contours=4,
                    )
                    axes[i].set_title(f"{band} ({low}-{high}Hz)", fontweight="bold")
                    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            plt.suptitle("Fig 10: Frequency Band Power", fontweight="bold", fontsize=16)
            plt.savefig(f"{fig_dir}/fig10_freq_bands.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Fig 10 Error: {e}")

        # --- 11. Split-Half ---
        try:
            mid = len(topo_maps) // 2
            half1 = np.mean(topo_maps[:mid], axis=0)
            half2 = np.mean(topo_maps[mid:], axis=0)
            corr = np.corrcoef(half1.flatten(), half2.flatten())[0, 1]
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))

            im1 = axes[0].imshow(half1, cmap="RdBu_r", origin="lower")
            im1.set_clip_path(self.make_circular_mask(axes[0], self.topo_map_size))
            axes[0].set_title("First Half (Mean)")
            axes[0].axis("off")

            im2 = axes[1].imshow(half2, cmap="RdBu_r", origin="lower")
            im2.set_clip_path(self.make_circular_mask(axes[1], self.topo_map_size))
            axes[1].set_title("Second Half (Mean)")
            axes[1].axis("off")

            sns.regplot(
                x=half1.flatten(),
                y=half2.flatten(),
                ax=axes[2],
                scatter_kws={"alpha": 0.1, "color": "k"},
                line_kws={"color": "red"},
            )
            axes[2].set_title(f"Correlation: r={corr:.3f}", fontweight="bold")
            plt.suptitle("Fig 11: Split-Half Reliability", fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{fig_dir}/fig11_split_half.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Fig 11 Error: {e}")

    def get_eeg_data(self, topo_maps, raw_dataset):
        # Normalize
        p1, p99 = np.percentile(topo_maps, [1, 99])
        topo_maps = np.clip(topo_maps, p1, p99)
        d_min, d_max = topo_maps.min(), topo_maps.max()
        if d_max - d_min > 0:
            topo_maps = (topo_maps - d_min) / (d_max - d_min)

        self.generate_research_figures(topo_maps, raw_dataset)

        data_torch = topo_maps.reshape(-1, 1, self.topo_map_size, self.topo_map_size)
        tensor_x = T.tensor(data_torch, dtype=T.float32)
        tensor_y = T.zeros(len(tensor_x))

        dataset = TensorDataset(tensor_x, tensor_y)
        train_len = int(0.7 * len(dataset))
        val_len = int(0.2 * len(dataset))
        test_len = len(dataset) - train_len - val_len

        train, val, test = random_split(dataset, [train_len, val_len, test_len])

        return VaeDatasets(
            DataLoader(
                train,
                batch_size=self.config["batch_size"],
                shuffle=True,
                drop_last=True,
            ),
            DataLoader(
                val, batch_size=self.config["batch_size"], shuffle=False, drop_last=True
            ),
            DataLoader(
                test,
                batch_size=self.config["batch_size"],
                shuffle=False,
                drop_last=True,
            ),
            train,
        )

    def process(self, participant_id=None):
        if participant_id:
            self.participant_id = participant_id
            self.config["data_path"] = f"{self.config['data_dir']}/{participant_id}.dat"
            self.config["output_path"] = (
                f"{self.config['data_dir']}/Topomaps/{participant_id}"
            )
            self.config["figure_dir"] = (
                f"{self.config['data_dir']}/Figure/{participant_id}"
            )
            os.makedirs(self.config["output_path"], exist_ok=True)
            os.makedirs(self.config["figure_dir"], exist_ok=True)

        self.logger.info(f"--- Processing {self.participant_id} ---")
        self.check_and_download_data()

        raw_data = self.load_preprocessed_dat(self.config["data_path"])
        if raw_data is None:
            raise ValueError("Data load failed")

        raw_mne = self.create_mne_raw_object(raw_data)
        topo_maps = self.generate_topographic_maps(raw_data, raw_mne)

        np.save(f"{self.config['output_path']}/topo_maps.npy", topo_maps)

        vae_datasets = self.get_eeg_data(topo_maps, raw_data)

        self.logger.info("Complete. Check the 'Figure' folder.")
        return vae_datasets
