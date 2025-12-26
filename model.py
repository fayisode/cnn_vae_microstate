import os
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import logging
import math as m
from matplotlib.colors import LinearSegmentedColormap
import mne
import warnings
from pathlib import Path


warnings.filterwarnings("ignore", message="The figure layout has changed to tight")

PI_TIMES_2 = torch.tensor(np.pi * 2.0)
EPSILON = 1e-8
gamma = 1e-3


class Encoder(nn.Module):
    def __init__(self, nc: int, ndf: int, latent_dim: int):
        super().__init__()
        self.nc = nc
        self.ndf = ndf
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ndf * 8, 1024, 1, 1, 0, bias=False),
            # nn.Conv2d(ndf * 8, 1024, 5, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, latent_dim)  # mu
        self.fc22 = nn.Linear(512, latent_dim)  # logvar

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for _m in self.modules():
            if isinstance(_m, nn.Conv2d):
                # Kaiming (He) initialization for layers followed by LeakyReLU
                # Accounts for the activation function's effect on variance
                nn.init.kaiming_normal_(
                    _m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(_m, nn.Linear):
                # Xavier initialization for linear layers
                # Maintains variance across layers for symmetric activations
                nn.init.xavier_normal_(_m.weight)
                if _m.bias is not None:
                    nn.init.constant_(_m.bias, 0)
            elif isinstance(_m, nn.BatchNorm2d):
                # Standard BatchNorm initialization
                nn.init.constant_(_m.weight, 1)
                nn.init.constant_(_m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv = self.encoder(x)
        h1 = self.fc1(conv.view(-1, 1024))
        return self.fc21(h1), self.fc22(h1)


class Decoder(nn.Module):
    def __init__(self, ngf: int, nc: int, latent_dim: int):
        super().__init__()
        self.ngf = ngf
        self.nc = nc
        self.latent_dim = latent_dim

        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, ngf * 8, 5, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ConvTranspose2d(ngf * 2, ngf * 2, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(ngf * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for _m in self.modules():
            if isinstance(_m, nn.ConvTranspose2d):
                # Kaiming initialization for transpose convolutions
                # Prevents checkerboard artifacts and maintains proper variance
                nn.init.kaiming_normal_(
                    _m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(_m, nn.Linear):
                # Xavier initialization for linear layers
                # Critical for latent space projection layers
                nn.init.xavier_normal_(_m.weight)
                if _m.bias is not None:
                    nn.init.constant_(_m.bias, 0)
            elif isinstance(_m, nn.BatchNorm2d):
                # Standard BatchNorm initialization
                # Ensures proper normalization from the start
                nn.init.constant_(_m.weight, 1)
                nn.init.constant_(_m.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        decoder_input = self.decoder_input(z)
        decoder_input = decoder_input.view(-1, 1024, 1, 1)
        # decoder_input = decoder_input.view(-1, 1024, 1, 1)
        return self.decoder(decoder_input)


#
# class Encoder(nn.Module):
#     def __init__(self, nc: int, ndf: int, latent_dim: int):
#         super().__init__()
#         self.nc = nc
#         self.ndf = ndf
#         self.latent_dim = latent_dim
#         self.encoder = nn.Sequential(
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.2),
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.2),
#             nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ndf * 8, 1024, 5, 1, 0, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc21 = nn.Linear(512, latent_dim)
#         self.fc22 = nn.Linear(512, latent_dim)
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         conv = self.encoder(x)
#         h1 = self.fc1(conv.view(-1, 1024))
#         return self.fc21(h1), self.fc22(h1)
#
#
# class Decoder(nn.Module):
#     def __init__(self, ngf: int, nc: int, latent_dim: int):
#         super().__init__()
#         self.ngf = ngf
#         self.nc = nc
#         self.latent_dim = latent_dim
#         self.decoder_input = nn.Sequential(
#             nn.Linear(latent_dim, 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 1024),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(1024, ngf * 8, 5, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.2),
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # nn.ConvTranspose2d(ngf * 2, ngf * 2, 3, 1, 1, bias=False),
#             # nn.BatchNorm2d(ngf * 2),
#             # nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.LeakyReLU(0.2, inplace=True),
#             # nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, bias=False),
#             # nn.BatchNorm2d(ngf),
#             # nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         decoder_input = self.decoder_input(z)
#         decoder_input = decoder_input.view(-1, 1024, 1, 1)
#         return self.decoder(decoder_input)


class LossBalancer:
    def __init__(
        self,
        init_beta: float = 0.001,
        min_beta: float = 0,
        max_beta: float = 0.3,
        beta_warmup_epochs: int = 30,
        adaptive_weight: bool = False,
        use_batch_cyclical: bool = False,
        n_cycles_per_epoch: int = 3,
        cycle_ratio: float = 0.5,
        gamma: float = 0.005,
    ):
        self.init_beta = init_beta
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.beta_warmup_epochs = beta_warmup_epochs
        self.adaptive_weight = adaptive_weight
        self.recon_avg: Optional[float] = None
        self.kld_avg: Optional[float] = None
        self.use_batch_cyclical = use_batch_cyclical
        self.n_cycles_per_epoch = n_cycles_per_epoch
        self.cycle_ratio = cycle_ratio
        self.gamma = gamma
        self.batch_schedule = None
        self.current_epoch = -1

    def get_state(self):
        return {
            "recon_avg": self.recon_avg,
            "kld_avg": self.kld_avg,
            "current_epoch": self.current_epoch,
        }

    def set_state(self, state):
        self.recon_avg = state.get("recon_avg")
        self.kld_avg = state.get("kld_avg")
        # self.current_epoch = state.get("current_epoch", -1)

    def setup_batch_cyclical_schedule(self, n_batches_per_epoch: int) -> np.ndarray:
        n_iter = n_batches_per_epoch
        n_cycle = self.n_cycles_per_epoch
        ratio = self.cycle_ratio
        start = self.gamma
        stop = self.gamma + 1
        L = np.ones(n_iter) * stop
        period = n_iter / n_cycle
        step = (stop - start) / (period * ratio)
        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i + c * period) < n_iter):
                if int(i + c * period) < n_iter:
                    L[int(i + c * period)] = v
                v += step
                i += 1
        L = np.clip(L, self.min_beta, self.max_beta)
        self.batch_schedule = L
        return L

    def get_beta(
        self,
        epoch: int,
        recon_loss: Union[torch.Tensor, float],
        kld_loss: Union[torch.Tensor, float],
        batch_idx: Optional[int] = None,
        n_batches_per_epoch: Optional[int] = None,
    ) -> float:

        if (
            self.use_batch_cyclical
            and batch_idx is not None
            and n_batches_per_epoch is not None
        ):
            return self.get_batch_cyclical_beta(
                epoch, batch_idx, n_batches_per_epoch, recon_loss, kld_loss
            )
        if epoch < self.beta_warmup_epochs:
            return self.init_beta + (self.max_beta - self.init_beta) * (
                epoch / self.beta_warmup_epochs
            )
        if not self.adaptive_weight:
            return self.max_beta
        if self.recon_avg is None:
            self.recon_avg = (
                recon_loss.detach().item()
                if isinstance(recon_loss, torch.Tensor)
                else recon_loss
            )
            self.kld_avg = (
                kld_loss.detach().item()
                if isinstance(kld_loss, torch.Tensor)
                else kld_loss
            )
        else:
            recon_value = (
                recon_loss.detach().item()
                if isinstance(recon_loss, torch.Tensor)
                else recon_loss
            )
            kld_value = (
                kld_loss.detach().item()
                if isinstance(kld_loss, torch.Tensor)
                else kld_loss
            )
            self.recon_avg = 0.9 * self.recon_avg + 0.1 * recon_value
            self.kld_avg = 0.9 * self.kld_avg + 0.1 * kld_value
        ratio = min(self.recon_avg / (self.kld_avg + EPSILON), 4.0)
        dynamic_beta = min(self.max_beta, max(self.min_beta, ratio))
        return dynamic_beta

    def get_batch_cyclical_beta(
        self,
        epoch: int,
        batch_idx: int,
        n_batches_per_epoch: int,
        recon_loss: Union[torch.Tensor, float],
        kld_loss: Union[torch.Tensor, float],
    ) -> float:
        if epoch != self.current_epoch:
            self.setup_batch_cyclical_schedule(n_batches_per_epoch)
            self.current_epoch = epoch
        safe_batch_idx = min(batch_idx, len(self.batch_schedule) - 1)
        base_beta = self.batch_schedule[safe_batch_idx]
        return float(base_beta)

    def cyclical_beta(
        self,
        epoch: int,
        total_epochs: int,
        n_cycles: int = 4,
        max_beta: Optional[float] = None,
    ) -> float:
        if max_beta is None:
            max_beta = self.max_beta
        cycle_length = max(1, total_epochs // n_cycles)
        cycle_progress = (epoch % cycle_length) / cycle_length
        beta = (
            self.min_beta
            + (max_beta - self.min_beta) * (1 - np.cos(np.pi * cycle_progress)) / 2
        )
        return beta


class MyModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        nClusters: int,
        batch_size: int,
        logger: any,
        device: torch.device,
        use_cyclical_annealing: bool = False,
        use_batch_cyclical: bool = True,
        nc: int = 1,
        ndf: int = 64,
        ngf: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.nClusters = nClusters
        self.nc = nc
        self.ndf = ndf
        self.ngf = ngf
        self.use_cyclical_annealing = use_cyclical_annealing
        self.use_batch_cyclical = use_batch_cyclical
        self.batch_size = batch_size
        self.flat_size = None
        self.device = device
        self.logger = logger
        self.loss_balancer = LossBalancer()
        self.encoder = Encoder(nc, ndf, latent_dim)
        self.decoder = Decoder(ngf, nc, latent_dim)
        self.pi_ = nn.Parameter(
            torch.log(torch.FloatTensor(self.nClusters).fill_(1) / self.nClusters),
            requires_grad=True,
        )
        self.prior_frozen = False
        self.mu_c = nn.Parameter(torch.randn(nClusters, latent_dim) * 2.0)
        self.log_var_c = nn.Parameter(torch.randn(nClusters, latent_dim) * 0.3)
        self.min_log_var = -4.0
        self._needs_data_init = True
        self.logger.info(
            f"📊 Trainer initialized with batch size: {batch_size} and latent dim: {latent_dim} and nClusters: {nClusters}"
        )

    def monitor_kl_health(self, mu, log_var):
        with torch.no_grad():
            kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kl_per_dim = kl_per_dim.mean(0)
            inactive_dims = (kl_per_dim < 0.01).sum().item()
            self.logger.info("KL per dimension stats:")
            self.logger.info(f"  Mean: {kl_per_dim.mean():.6f}")
            self.logger.info(f"  Std: {kl_per_dim.std():.6f}")
            self.logger.info(f"  Min: {kl_per_dim.min():.6f}")
            self.logger.info(f"  Max: {kl_per_dim.max():.6f}")
            self.logger.info(
                f"  Inactive dimensions (KL < 0.01): {inactive_dims}/{len(kl_per_dim)}"
            )
            if inactive_dims > len(kl_per_dim) * 0.5:
                self.logger.warning(
                    "⚠️  More than 50% of latent dimensions are inactive!"
                )
                return False
            return True

    def detect_posterior_collapse(self, data_loader, threshold=0.1):
        self.eval()
        total_kl = 0
        total_mixture_kl = 0
        num_samples = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i > 10:
                    break
                data = data.to(self.device)
                # data = (data - data.min()) / (data.max() - data.min() + EPSILON)
                mu, log_var = self.encode(data)
                standard_kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                total_kl += standard_kl.item()
                mixture_kl = self.KLD(mu, log_var, normalize=False)
                total_mixture_kl += mixture_kl.item()
                num_samples += data.size(0)
        avg_standard_kl = total_kl / num_samples
        avg_mixture_kl = total_mixture_kl / num_samples
        self.logger.info("Collapse Detection:")
        self.logger.info(f"  Standard KL per sample: {avg_standard_kl:.6f}")
        self.logger.info(f"  Mixture KL per sample: {avg_mixture_kl:.6f}")
        if avg_standard_kl < threshold or avg_mixture_kl < threshold:
            self.logger.warning("🚨 POSTERIOR COLLAPSE DETECTED!")
            self.logger.warning(f"   Standard KL: {avg_standard_kl:.6f} < {threshold}")
            self.logger.warning(f"   Mixture KL: {avg_mixture_kl:.6f} < {threshold}")
            self.train()
            return True
        self.train()
        return False

    def initialize_from_data(self, train_loader):
        if hasattr(self, "_needs_data_init") and self._needs_data_init:
            self.logger.info("Initializing clusters from data distribution...")
            self.eval()
            latents = []
            with torch.no_grad():
                for i, (data, _) in enumerate(train_loader):
                    if i > 100:
                        break
                    data = data.to(self.device)
                    # data = (data - data.min()) / (data.max() - data.min() + EPSILON)
                    mu, _ = self.encode(data)
                    latents.append(mu.cpu())
            if latents:
                latents = torch.cat(latents, 0).numpy()
                self.logger.info(
                    f"Collected {len(latents)} latent samples for initialization"
                )
                kmeans = KMeans(
                    n_clusters=self.nClusters, n_init=20, max_iter=500, random_state=42
                )
                cluster_labels = kmeans.fit_predict(latents)
                mu_c = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
                log_var_c = torch.zeros(self.nClusters, self.latent_dim)
                for k in range(self.nClusters):
                    cluster_data = latents[cluster_labels == k]
                    if len(cluster_data) > 1:
                        cluster_var = np.var(cluster_data, axis=0)
                        cluster_var = np.maximum(cluster_var, 1.0)
                        log_var_c[k] = torch.log(
                            torch.tensor(cluster_var, dtype=torch.float32)
                        )
                    else:
                        log_var_c[k] = torch.full((self.latent_dim,), 0.0)
                self.mu_c.data = mu_c.to(self.device)
                self.log_var_c.data = log_var_c.to(self.device)
                self._needs_data_init = False
                self.logger.info("✅ Clusters initialized from data distribution")
            self.train()

    def check_prior_health(self):
        weights = torch.exp(self.pi_).detach().cpu().numpy()
        self.logger.info(f"Cluster weights: {weights}")
        self.logger.info(f"Cluster means std: {self.mu_c.std().item():.6f}")
        self.logger.info(
            f"Log var range: [{self.log_var_c.min().item():.3f}, {self.log_var_c.max().item():.3f}]"
        )
        if np.max(weights) > 0.95:
            self.logger.warning("One cluster dominates (weight > 95%)")
        if self.mu_c.std() < 0.01:
            self.logger.warning("Cluster means are too similar")

    def check_latent_space_health(self, data_loader):
        self.eval()
        all_mu = []
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i > 10:
                    break
                data = data.to(self.device)
                # data = (data - data.min()) / (data.max() - data.min() + EPSILON)
                mu, _ = self.encode(data)
                all_mu.append(mu.cpu())
        if all_mu:
            all_mu = torch.cat(all_mu, 0)
            latent_std = all_mu.std(0).mean().item()
            self.logger.info(f"Average latent dimension std: {latent_std:.6f}")
            if latent_std < 0.01:
                self.logger.warning("Latent space has collapsed (low variance)")
                return False
        return True

    def cluster_separation_loss(self):
        try:
            mu_c = self.mu_c
            if torch.isnan(mu_c).any():
                self.logger.warning("NaN detected in cluster means")
                mu_c = torch.nan_to_num(mu_c, nan=0.0)
                self.mu_c.data = mu_c
            distances = torch.cdist(mu_c, mu_c)
            distances = torch.nan_to_num(distances, nan=0.0)
            mask = torch.eye(self.nClusters, device=self.device).bool()
            distances = distances.masked_fill(mask, float("inf"))
            min_dist = distances.min()
            if torch.isnan(min_dist) or torch.isinf(min_dist):
                min_dist = torch.tensor(2.0, device=self.device)
            separation_loss = torch.relu(2.0 - min_dist)
            separation_loss = torch.nan_to_num(separation_loss, nan=0.0)
            return separation_loss
        except Exception as e:
            self.logger.warning(f"Cluster separation loss computation failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        if torch.isnan(mu).any() or torch.isnan(std).any():
            self.logger.warning("NaN detected in reparameterize inputs")
            mu = torch.nan_to_num(mu, nan=0.0)
            std = torch.nan_to_num(std, nan=1.0)
        eps = torch.randn_like(std)
        z = eps * std + mu
        z = torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
        return z

    def gaussian_pdf_log(
        self, x: torch.Tensor, mu: torch.Tensor, log_sigma2: torch.Tensor
    ) -> torch.Tensor:
        log_sigma2 = torch.clamp(log_sigma2, min=-10, max=10)
        if (
            torch.isnan(x).any()
            or torch.isnan(mu).any()
            or torch.isnan(log_sigma2).any()
        ):
            self.logger.warning("NaN detected in gaussian_pdf_log inputs")
            x = torch.nan_to_num(x, nan=0.0)
            mu = torch.nan_to_num(mu, nan=0.0)
            log_sigma2 = torch.nan_to_num(log_sigma2, nan=-1.0)
        diff = x - mu
        sigma2 = torch.exp(log_sigma2) + EPSILON
        log_prob = -0.5 * (np.log(2 * np.pi) + log_sigma2 + (diff**2) / sigma2)
        log_prob = torch.sum(log_prob, dim=1)
        log_prob = torch.nan_to_num(log_prob, nan=-1000.0, posinf=0.0, neginf=-1000.0)
        log_prob = torch.clamp(log_prob, min=-1000.0, max=100.0)
        return log_prob

    def gaussian_pdfs_log(
        self, x: torch.Tensor, mus: torch.Tensor, log_sigma2s: torch.Tensor
    ) -> torch.Tensor:
        G = []
        for c in range(self.nClusters):
            G.append(
                self.gaussian_pdf_log(
                    x, mus[c : c + 1, :], log_sigma2s[c : c + 1, :]
                ).view(-1, 1)
            )
        return torch.cat(G, 1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            mu, logvar = self.encoder(x)
            if torch.isnan(mu).any() or torch.isnan(logvar).any():
                self.logger.warning("NaN detected in encoder output")
                mu = torch.nan_to_num(mu, nan=0.0)
                logvar = torch.nan_to_num(logvar, nan=-1.0)
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)
            z = self.reparameterize(mu, logvar)
            z = torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
            reconstructed_x = self.decoder(z)
            if torch.isnan(reconstructed_x).any():
                self.logger.warning("NaN detected in decoder output")
                reconstructed_x = torch.nan_to_num(reconstructed_x, nan=0.5)
            reconstructed_x = torch.clamp(reconstructed_x, 0.0, 1.0)
            return reconstructed_x, mu, logvar
        except Exception as e:
            self.logger.warning(f"Forward pass failed: {e}")
            batch_size = x.size(0)
            mu = torch.zeros(batch_size, self.latent_dim, device=x.device)
            logvar = torch.full((batch_size, self.latent_dim), -1.0, device=x.device)
            reconstructed_x = torch.zeros_like(x)
            return reconstructed_x, mu, logvar

    # def predict_robust(self, x: torch.Tensor) -> np.ndarray:
    #     was_training = self.training
    #     self.eval()
    #
    #     try:
    #         with torch.no_grad():
    #             z_mu, z_log_var = self.encode(x)
    #
    #             # Handle invalid values defensively
    #             if torch.isnan(z_mu).any() or torch.isnan(z_log_var).any():
    #                 self.logger.warning("NaN detected in encoder output")
    #                 z_mu = torch.nan_to_num(z_mu, nan=0.0)
    #                 z_log_var = torch.nan_to_num(z_log_var, nan=-1.0)
    #
    #             # Multiple sampling with majority vote (robust approach)
    #             n_samples = 5
    #             predictions = []
    #
    #             for _ in range(n_samples):
    #                 # Correct sampling using your reparameterize method
    #                 z = self.reparameterize(z_mu, z_log_var)
    #                 z = torch.nan_to_num(z, nan=0.0)
    #
    #                 # Correct probability calculation (no double-log!)
    #                 log_pi = self.pi_  # Already in log space
    #                 log_probs = self.gaussian_pdfs_log(z, self.mu_c, self.log_var_c)
    #
    #                 # Clean probability computation
    #                 log_yita_c = log_pi.unsqueeze(0) + log_probs
    #                 yita_c = torch.exp(log_yita_c)
    #                 yita_c = yita_c + 1e-8  # Numerical stability
    #                 yita_c = yita_c / yita_c.sum(1, keepdim=True)  # Normalize
    #
    #                 pred = torch.argmax(yita_c, dim=1)
    #                 predictions.append(pred.cpu().numpy())
    #
    #             # Majority vote across all samples
    #             predictions = np.array(predictions)  # Shape: (n_samples, batch_size)
    #             final_pred = []
    #
    #             for i in range(predictions.shape[1]):  # For each sample in batch
    #                 sample_predictions = predictions[
    #                     :, i
    #                 ]  # All predictions for this sample
    #                 values, counts = np.unique(sample_predictions, return_counts=True)
    #                 most_common_pred = values[np.argmax(counts)]  # Majority vote
    #                 final_pred.append(most_common_pred)
    #
    #             return np.array(final_pred)
    #
    #     except Exception as e:
    #         self.logger.error(f"Critical prediction error: {e}")
    #         # Fallback: return random predictions
    #         return np.random.randint(0, self.nClusters, size=x.size(0))
    #     finally:
    #         if was_training:
    #             self.train()

    # def predict_robust(self, x: torch.Tensor) -> np.ndarray:
    #     was_training = self.training
    #     self.eval()
    #     try:
    #         with torch.no_grad():
    #             z_mu, z_log_var = self.encode(x)
    #             invalid_mask = (
    #                 torch.isnan(z_mu)
    #                 | torch.isinf(z_mu)
    #                 | torch.isnan(z_log_var)
    #                 | torch.isinf(z_log_var)
    #             )
    #             if invalid_mask.any():
    #                 self.logger.warning(
    #                     "NaN/Inf detected in encoder output; handling per sample"
    #                 )
    #                 z_mu = torch.where(invalid_mask, torch.zeros_like(z_mu), z_mu)
    #             if (
    #                 torch.isnan(self.mu_c).any()
    #                 or torch.isinf(self.mu_c).any()
    #                 or torch.isnan(self.log_var_c).any()
    #                 or torch.isinf(self.log_var_c).any()
    #                 or torch.isnan(self.pi_).any()
    #                 or torch.isinf(self.pi_).any()
    #             ):
    #                 self.logger.error("Invalid cluster parameters - NaN/Inf detected")
    #                 return np.random.randint(0, self.nClusters, size=x.size(0))
    #             z = z_mu
    #             log_pi = self.pi_
    #             log_probs = self.gaussian_pdfs_log(z, self.mu_c, self.log_var_c)
    #             if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
    #                 self.logger.error("Invalid log probabilities")
    #                 return np.random.randint(0, self.nClusters, size=x.size(0))
    #             log_yita_c = log_pi.unsqueeze(0) + log_probs
    #             log_yita_c = log_yita_c - torch.logsumexp(
    #                 log_yita_c, dim=1, keepdim=True
    #             )
    #             yita_c = torch.exp(log_yita_c)
    #             if torch.isnan(yita_c).any() or torch.isinf(yita_c).any():
    #                 # self.logger.error("Invalid responsibilities")
    #                 return np.random.randint(0, self.nClusters, size=x.size(0))
    #             pred = torch.argmax(yita_c, dim=1)
    #             self.logger.debug(
    #                 f"Mean responsibilities: {yita_c.mean(dim=0).cpu().numpy()}"
    #             )
    #             return pred.cpu().numpy()
    #     except Exception as e:
    #         self.logger.error(f"Critical prediction error: {e}")
    #         return np.random.randint(0, self.nClusters, size=x.size(0))
    #     finally:
    #         if was_training:
    #             self.train()

    def predict_robust(self, x: torch.Tensor) -> np.ndarray:
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                z_mu, z_log_var = self.encode(x)

                # Check for problematic values
                if torch.isnan(z_mu).any() or torch.isnan(z_log_var).any():
                    self.logger.warning(
                        "NaN in latent variables - using fallback prediction"
                    )
                    return np.random.randint(0, self.nClusters, size=x.size(0))

                # Use mean instead of sampling for more stable predictions
                z = z_mu  # Use mean directly instead of sampling

                log_pi = self.pi_
                log_sigma2_c = self.log_var_c
                mu_c = self.mu_c

                # Check cluster parameters
                if torch.isnan(mu_c).any() or torch.isnan(log_sigma2_c).any():
                    self.logger.warning("NaN in cluster parameters")
                    return np.random.randint(0, self.nClusters, size=x.size(0))

                log_probs = self.gaussian_pdfs_log(z, mu_c, log_sigma2_c)
                log_yita_c = log_pi.unsqueeze(0) + log_probs

                # More stable softmax
                log_yita_c = log_yita_c - torch.logsumexp(
                    log_yita_c, dim=1, keepdim=True
                )
                yita_c = torch.exp(log_yita_c)

                pred = torch.argmax(yita_c, dim=1)
                return pred.cpu().numpy()

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return np.random.randint(0, self.nClusters, size=x.size(0))
        finally:
            if was_training:
                self.train()

    # def predict_robust(self, x: torch.Tensor) -> np.ndarray:
    #     was_training = self.training
    #     self.eval()
    #     with torch.no_grad():
    #         z_mu, z_log_var = self.encode(x)
    #         n_samples = 5
    #         predictions = []
    #         for _ in range(n_samples):
    #             z = torch.randn_like(z_mu) * torch.exp(z_log_var / 2) + z_mu
    #             z = torch.nan_to_num(z, nan=0)
    #             log_pi = self.pi_
    #             log_sigma2_c = self.log_var_c
    #             mu_c = self.mu_c
    #             yita_c = torch.exp(
    #                 log_pi.unsqueeze(0) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c)
    #             )
    #             yita_c = yita_c + 1e-8
    #             yita_c = yita_c / yita_c.sum(1, keepdim=True)
    #             pred = torch.argmax(yita_c, dim=1)
    #             predictions.append(pred.cpu().numpy())
    #         predictions = np.array(predictions)
    #         final_pred = []
    #         for i in range(predictions.shape[1]):
    #             values, counts = np.unique(predictions[:, i], return_counts=True)
    #             final_pred.append(values[np.argmax(counts)])
    #         final_pred = np.array(final_pred)
    #     if was_training:
    #         self.train()
    #     return final_pred

    def predict(self, x: torch.Tensor) -> np.ndarray:

        # with torch.no_grad():
        #     # Encode data to get latent representations
        #     mu, _ = self.encode(x)
        #
        #     # Move to CPU and convert to numpy
        #     latents = mu.cpu().numpy()
        #
        #     # Apply K-means clustering
        #     from sklearn.cluster import KMeans
        #
        #     kmeans = KMeans(n_clusters=self.nClusters, random_state=42, n_init=10)
        #     clusters = kmeans.fit_predict(latents)
        clusters = self.predict_robust(x)

        # try:
        #     with torch.no_grad():
        #         # Get latent representations for silhouette calculation
        #         mu, _ = self.encode(x)
        #         latents = mu.cpu().numpy()
        #
        #         # Check if we have enough clusters for silhouette calculation
        #         unique_clusters = np.unique(clusters)
        #         if len(unique_clusters) >= 2 and len(clusters) >= 2:
        #             from sklearn.metrics import silhouette_score
        #
        #             silhouette = silhouette_score(latents, clusters)
        #             self.logger.info(f"    Silhouette score: {silhouette:.4f}")
        #         else:
        #             self.logger.info(
        #                 f"    Silhouette score: N/A (only {len(unique_clusters)} clusters)"
        #             )
        #
        # except Exception as e:
        #     self.logger.warning(f"    Silhouette calculation failed: {e}")

        return clusters
        # return self.predict_robust(x)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def RE(
        self, recon_x: torch.Tensor, x: torch.Tensor, normalize: bool = False
    ) -> torch.Tensor:
        recon_x = torch.clamp(recon_x, 0, 1)
        x = torch.clamp(x, 0, 1)
        flat_size = recon_x[0].numel()
        self.flat_size = flat_size
        mse_loss = torch.nn.functional.mse_loss(
            recon_x.view(-1, flat_size),
            x.view(-1, flat_size),
            reduction="sum",
        )
        # if True:
        #     batch_size = x.size(0)
        #     return mse_loss / (batch_size * flat_size)
        return mse_loss

    def KLD(
        self, mu: torch.Tensor, log_var: torch.Tensor, normalize: bool = False
    ) -> torch.Tensor:
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        log_var_c = torch.clamp(self.log_var_c, min=self.min_log_var, max=2.0)
        pi = self.pi_
        mu_c = self.mu_c
        z = self.reparameterize(mu, log_var)
        log_pi = pi.unsqueeze(0)
        log_gaussian = self.gaussian_pdfs_log(z, mu_c, log_var_c)
        log_yita_c = log_pi + log_gaussian
        log_sum = torch.logsumexp(log_yita_c, dim=1, keepdim=True)
        log_yita_c = log_yita_c - log_sum
        yita_c = torch.exp(log_yita_c)
        kl_first_term = 0.5 * torch.mean(
            torch.sum(
                yita_c
                * torch.sum(
                    log_var_c.unsqueeze(0)
                    + torch.exp(log_var.unsqueeze(1) - log_var_c.unsqueeze(0))
                    + (mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2)
                    / (torch.exp(log_var_c.unsqueeze(0)) + EPSILON),
                    2,
                ),
                1,
            )
        )
        kl_second_term = torch.mean(torch.sum(yita_c * (log_yita_c - log_pi), 1))
        entropy_term = 0.5 * torch.mean(torch.sum(1 + log_var, 1))
        loss = kl_first_term + kl_second_term - entropy_term
        loss = torch.max(loss, torch.tensor(EPSILON, device=loss.device))
        # loss = torch.clamp(loss, min=0.1 * mu.size(1))
        # loss = torch.clamp(loss)
        # if True:
        #     latent_dim = mu.size(1)
        #     batch_size = mu.size(0)
        #     return loss / (batch_size * latent_dim)
        return loss

    def KLD_with_free_bits(self, mu, log_var, free_bits_per_dim=0.5):
        mixture_kld = self.KLD(mu, log_var, normalize=False)
        standard_kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        standard_kl = torch.clamp(standard_kl, min=free_bits_per_dim)
        standard_kl_total = torch.sum(standard_kl, dim=1).mean()
        return torch.max(mixture_kld, standard_kl_total)

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        epoch: int = 0,
        total_epochs: int = 100,
        normalize: bool = True,
        batch_idx: int = 0,
        n_batches_per_epoch: int = 1000,
        is_pretraining: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalize = True
        try:
            reconst_loss = self.RE(recon_x, x, normalize=False)
            if torch.isnan(reconst_loss) or torch.isinf(reconst_loss):
                reconst_loss = torch.tensor(
                    1000.0, device=recon_x.device, requires_grad=True
                )
            kld_loss = self.KLD(mu, log_var, normalize=normalize)
            if torch.isnan(kld_loss) or torch.isinf(kld_loss):
                kld_loss = torch.tensor(100.0, device=mu.device, requires_grad=True)
            if self.use_batch_cyclical:
                beta = self.loss_balancer.get_beta(
                    epoch,
                    reconst_loss,
                    kld_loss,
                    batch_idx=batch_idx,
                    n_batches_per_epoch=n_batches_per_epoch,
                )
            elif self.use_cyclical_annealing:
                beta = self.loss_balancer.cyclical_beta(epoch, total_epochs, n_cycles=4)
            else:
                beta = self.loss_balancer.get_beta(epoch, reconst_loss, kld_loss)

            # beta = max(0.001, min(beta, 10.0))
            total_loss = reconst_loss + kld_loss * beta
            total_loss = torch.clamp(total_loss, min=EPSILON, max=10000.0)
        except Exception as e:
            self.logger.warning(f"Loss computation failed: {e}")
            reconst_loss = torch.tensor(
                1000.0, device=recon_x.device, requires_grad=True
            )
            kld_loss = torch.tensor(100.0, device=mu.device, requires_grad=True)
            total_loss = torch.tensor(1100.0, device=recon_x.device, requires_grad=True)
        return reconst_loss, kld_loss, total_loss

    # Add these two new methods to your MyModel class

    def freeze_prior(self) -> None:
        for param in [self.pi_, self.mu_c, self.log_var_c]:
            param.requires_grad = False
        self.prior_frozen = True

    def unfreeze_prior(self) -> None:
        for param in [self.pi_, self.mu_c, self.log_var_c]:
            param.requires_grad = True
        self.prior_frozen = False

    def gradual_unfreeze(self, epoch: int) -> None:
        if epoch >= 2:
            self.pi_.requires_grad = True
        if epoch >= 3:
            self.mu_c.requires_grad = True
            self.log_var_c.requires_grad = True
            self.prior_frozen = False

    def _apply_circular_mask_single_torch(self, image, radius_factor=0.95):
        if image.device.type == "cuda":
            image = image.cpu()
        height, width = image.shape
        center_y, center_x = height // 2, width // 2
        radius = int(min(center_x, center_y) * radius_factor)
        y = (
            torch.arange(height, device="cpu")
            .float()
            .unsqueeze(1)
            .expand(height, width)
        )
        x = torch.arange(width, device="cpu").float().unsqueeze(0).expand(height, width)
        dist = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        mask = (dist <= radius).float()
        return image * mask

    def get_cluster_centroids_and_visualize(self, output_dir: str = "images/clusters"):
        os.makedirs(output_dir, exist_ok=True)
        latent_centroids = self.mu_c.detach().cpu()
        self.eval()
        with torch.no_grad():
            decoded_centroids = self.decode(self.mu_c)
        decoded_centroids_np = decoded_centroids.detach().cpu().numpy()
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
        eeg_info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types="eeg")
        montage = mne.channels.make_standard_montage("biosemi32")
        eeg_info.set_montage(montage)
        channel_types = [None, "mag", "eeg", "grad"]
        colormap_variants = {
            "standard": LinearSegmentedColormap.from_list(
                "custom_eeg_cmap",
                [(0, 0, 0.8), (0, 1, 0.5), (1, 1, 0.8), (1, 0.5, 0), (0.8, 0, 0)],
                N=100,
            ),
            "Spectral_r": "Spectral_r",
            "RdBu_r": "RdBu_r",
            "viridis": "viridis",
            "plasma": "plasma",
            "coolwarm": "coolwarm",
            "jet": "jet",
            "rainbow": "rainbow",
        }
        interp_methods = ["cubic", "linear", "nearest"]
        img_shape = decoded_centroids_np[0].shape
        self.logger.info(f"Decoded centroid shape: {img_shape}")
        if len(img_shape) == 3:
            channels, height, width = img_shape
            self.logger.info(f"Processing multi-channel image with {channels} channels")
        else:
            height, width = img_shape
            self.logger.info(
                f"Processing single-channel image of size {height}x{width}"
            )
        center_x, center_y = width // 2, height // 2
        if abs(height - width) > width * 0.1:
            self.logger.warning(
                f"Image is not square ({width}x{height}), which may affect visualization accuracy"
            )
        channel_values_all = []
        for i in range(self.nClusters):
            cluster_dir = os.path.join(output_dir, f"cluster_{i+1}")
            os.makedirs(cluster_dir, exist_ok=True)
            centroid_img = decoded_centroids_np[i]
            if len(centroid_img.shape) == 3:
                if centroid_img.shape[0] == 1:
                    centroid_img = centroid_img.squeeze(0)
                    self.logger.info("Using single channel from decoded centroid")
                else:
                    self.logger.warning(
                        f"Multiple channels ({centroid_img.shape[0]}) detected in decoded centroid. Using the first channel by default."
                    )
                    centroid_img = centroid_img[0]
            pos_3d = []
            valid_ch_names = []
            for ch_name in ch_names:
                if ch_name in montage.get_positions()["ch_pos"]:
                    ch_pos = montage.get_positions()["ch_pos"][ch_name]
                    pos_3d.append(ch_pos)
                    valid_ch_names.append(ch_name)
                else:
                    self.logger.warning(
                        f"Channel {ch_name} not found in montage positions"
                    )
            self.logger.info(
                f"Found {len(pos_3d)}/{len(ch_names)} channel positions in montage"
            )

            def _azim_proj(pos):
                [r, elev, az] = _cart2sph(pos[0], pos[1], pos[2])
                return _pol2cart(az, m.pi / 2 - elev)

            def _cart2sph(x, y, z):
                x2_y2 = x**2 + y**2
                r = m.sqrt(x2_y2 + z**2)
                elev = m.atan2(z, m.sqrt(x2_y2))
                az = m.atan2(y, x)
                return r, elev, az

            def _pol2cart(theta, rho):
                return rho * m.cos(theta), rho * m.sin(theta)

            pos_2d = []
            for pos in pos_3d:
                pos_2d.append(_azim_proj(pos))
            pos_2d = np.array(pos_2d)
            min_x, min_y = np.min(pos_2d, axis=0)
            max_x, max_y = np.max(pos_2d, axis=0)
            self.logger.info(
                f"Electrode position range: X: [{min_x:.2f}, {max_x:.2f}], Y: [{min_y:.2f}, {max_y:.2f}]"
            )
            channel_values = np.zeros(len(valid_ch_names))
            channel_value_map = {}
            for idx, ((x, y), ch_name) in enumerate(zip(pos_2d, valid_ch_names)):
                norm_x = (x - min_x) / (max_x - min_x) * 2 - 1
                norm_y = (y - min_y) / (max_y - min_y) * 2 - 1
                radius = min(center_x, center_y) * 0.95
                img_x = int(center_x + norm_x * radius)
                img_y = int(center_y - norm_y * radius)
                img_x = max(0, min(img_x, width - 1))
                img_y = max(0, min(img_y, height - 1))
                if idx < 5:
                    self.logger.info(
                        f"Electrode {ch_name}: Normalized ({norm_x:.2f}, {norm_y:.2f}) -> Image coordinates ({img_x}, {img_y})"
                    )
                channel_values[idx] = centroid_img[img_y, img_x]
                channel_value_map[ch_name] = channel_values[idx]
            value_examples = {
                k: channel_value_map[k] for k in list(channel_value_map.keys())[:5]
            }
            self.logger.info(f"Sample extracted values: {value_examples}")
            if np.any(np.isnan(channel_values)):
                self.logger.warning("Found NaN values in extracted channel values")
                channel_values = np.nan_to_num(channel_values, nan=0.0)
            original_min = np.min(channel_values)
            original_max = np.max(channel_values)
            if original_max != original_min:
                channel_values = (channel_values - original_min) / (
                    original_max - original_min
                ) * 2 - 1
                self.logger.info(
                    f"Normalized channel values from range [{original_min:.6f}, {original_max:.6f}] to [-1, 1]"
                )
            else:
                self.logger.warning(
                    f"All channel values are identical ({original_min}). This may indicate an issue with value extraction."
                )
            channel_values_all.append(channel_values)
            for ch_type in channel_types:
                for cmap_name, cmap in colormap_variants.items():
                    for interp in interp_methods:
                        if interp == "nearest" and cmap_name not in [
                            "standard",
                            "viridis",
                            "RdBu_r",
                        ]:
                            continue
                        plt.figure(figsize=(6, 6), facecolor="white")
                        contours = 0 if interp == "nearest" else 7
                        mne.viz.plot_topomap(
                            channel_values,
                            eeg_info,
                            axes=plt.gca(),
                            ch_type=ch_type,
                            cmap=cmap,
                            contours=contours,
                            outlines="head",
                            sensors=True,
                            res=128,
                            extrapolate="head",
                            image_interp=interp,
                            show=False,
                        )
                        ch_type_str = ch_type if ch_type else "default"
                        title = plt.title(
                            f"Cluster {i+1} - {ch_type_str} - {cmap_name} - {interp}",
                            fontsize=12,
                        )
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(
                                cluster_dir,
                                f"cluster_{i+1}_{ch_type_str}_{cmap_name}_{interp}.png",
                            ),
                            dpi=300,
                            bbox_inches="tight",
                            facecolor="white",
                        )
                        plt.close()
        results = {
            "latent_centroids": latent_centroids.numpy(),
            "decoded_centroids": decoded_centroids_np,
            "channel_values": np.array(channel_values_all),
        }
        self.logger.info(
            f"Visualization complete. Generated comprehensive visualizations for {self.nClusters} clusters in {output_dir}"
        )
        return results

    def pretrain(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        train_set: torch.utils.data.Dataset,
        epochs: int = 30,
        gamma_steps: int = 1000,
        initial_gamma: float = 0.1,
        freeze_prior_epochs: int = 0,
        evaluate_every: int = 5,
        device: Optional[torch.device] = None,
        output_dir: Optional[Path] = None,
    ) -> dict:
        if device is None:
            device = next(self.parameters()).device

        pretrain_checkpoint_path = None
        if output_dir:
            pretrain_checkpoint_path = output_dir / "pretrain_checkpoint.pth"

        start_step = 0
        start_epoch = 0
        phase = "gamma"
        history = {
            "epoch_losses": [],
            "reconstruct_losses": [],
            "kld_losses": [],
            "nmi_scores": [],
            "ari_scores": [],
            "silhouette_scores": [],
            "db_scores": [],
            "ch_scores": [],
            "beta_values": [],
        }
        collapse_counter = 0

        if pretrain_checkpoint_path and pretrain_checkpoint_path.exists():
            self.logger.info(f"Resuming pretraining from {pretrain_checkpoint_path}")
            checkpoint = torch.load(
                pretrain_checkpoint_path, map_location=device, weights_only=False
            )
            self.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            phase = checkpoint.get("phase", "gamma")
            start_step = checkpoint.get("step_count", 0)
            start_epoch = checkpoint.get("epoch", 0)
            history = checkpoint.get("history", history)
            collapse_counter = checkpoint.get("collapse_counter", 0)
            if "loss_balancer_state" in checkpoint:
                self.loss_balancer.set_state(checkpoint["loss_balancer_state"])
            self.logger.info(
                f"Resuming from phase '{phase}', step {start_step}, epoch {start_epoch}"
            )

        self.train()

        self.initialize_from_data(train_loader)

        if freeze_prior_epochs > 0 and start_epoch < freeze_prior_epochs:
            self.freeze_prior()
            self.logger.info(f"Prior frozen for {freeze_prior_epochs} epochs")
        else:
            self.logger.info("Prior NOT frozen - training with all parameters active")

        step_count = start_step if phase == "gamma" else gamma_steps
        n_batches_per_epoch = len(train_loader)

        if phase == "gamma":
            self.logger.info(f"Starting/Resuming gamma steps from step {start_step}")
            gamma_progress_bar = tqdm(
                total=gamma_steps, initial=start_step, desc="Gamma Steps"
            )

            gamma_complete = False
            while not gamma_complete:
                for batch_idx, (data, _) in enumerate(train_loader):
                    if step_count < start_step:
                        if (batch_idx + 1) * data.size(0) > step_count:
                            pass
                        else:
                            continue

                    if step_count >= gamma_steps:
                        gamma_complete = True
                        break

                    data = data.to(device)
                    # data = (data - data.min()) / (data.max() - data.min() + EPSILON)
                    optimizer.zero_grad()
                    recon_batch, mu, logvar = self(data)
                    # re_loss, kld_loss, loss = self.loss_function(
                    #     recon_batch,
                    #     data,
                    #     mu,
                    #     logvar,
                    #     normalize=True,
                    #     epoch=0,
                    #     total_epochs=epochs + 1,
                    #     batch_idx=batch_idx,
                    #     n_batches_per_epoch=n_batches_per_epoch,
                    #     is_pretraining=True,
                    # )
                    # Simple loss with fixed gamma (no complex balancing)
                    re_loss = self.RE(recon_batch, data)
                    kld_loss = self.KLD(mu, logvar)
                    loss = re_loss + (1e-3 * kld_loss)  # Fixed gamma!
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()

                    if step_count % 50 == 0:
                        kl_healthy = self.monitor_kl_health(mu, logvar)
                        if not kl_healthy:
                            collapse_counter += 1
                        # current_beta = (
                        #     self.loss_balancer.get_beta(
                        #         0, re_loss, kld_loss, batch_idx, n_batches_per_epoch
                        #     )
                        #     if self.use_batch_cyclical
                        #     else self.loss_balancer.get_beta(0, re_loss, kld_loss)
                        # )
                        current_lr = optimizer.param_groups[0]["lr"]
                        self.logger.info(
                            f"\n{'='*80}\n"
                            f"GAMMA STEP {step_count:4d} | LR: {current_lr:.2e}\n"
                            f"{'='*80}\n"
                            f"🔄 Loss Components:\n"
                            f"   └─ Reconstruction: {re_loss.item():8.6f}\n"
                            f"   └─ KLD:           {kld_loss.item():8.6f}\n"
                            f"   └─ Total:         {loss.item():8.6f}\n"
                            f"   └─ Beta (γ):      {1e-5:8.6f}\n"
                            f"📊 Health Status:\n"
                            f"   └─ KL Healthy:    {'✅ Yes' if kl_healthy else '❌ No'}\n"
                            f"   └─ Collapse Count: {collapse_counter}\n"
                            f"{'='*80}\n"
                        )

                    history["reconstruct_losses"].append(re_loss.item())
                    history["kld_losses"].append(kld_loss.item())
                    history["epoch_losses"].append(loss.item())

                    step_count += 1
                    gamma_progress_bar.update(1)

                    if pretrain_checkpoint_path and step_count % 50 == 0:
                        torch.save(
                            {
                                "phase": "gamma",
                                "step_count": step_count,
                                "epoch": 0,
                                "model_state_dict": self.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "history": history,
                                "collapse_counter": collapse_counter,
                                "loss_balancer_state": self.loss_balancer.get_state(),
                            },
                            pretrain_checkpoint_path,
                        )

                if not gamma_complete:  # Reset dataloader for next pass if needed
                    self.logger.info("Restarting dataloader to continue gamma steps...")

            gamma_progress_bar.close()
            phase = "epochs"
            start_epoch = 0

        if self.detect_posterior_collapse(train_loader, threshold=0.1):
            self.logger.warning(
                "🚨 Collapse detected after gamma steps - adjusting parameters"
            )
            self.loss_balancer.min_beta *= 0.1
            self.loss_balancer.max_beta *= 0.5
            self.loss_balancer.gamma *= 0.5
            collapse_counter += 1

        for epoch in range(start_epoch, freeze_prior_epochs):
            self.gradual_unfreeze(epoch)
            epoch_loss, reconstruct_loss, kld_loss_total = 0, 0, 0
            epoch_betas = []
            sampled_mu, sampled_logvar = [], []
            for batch_idx, (data, _) in enumerate(
                tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}")
            ):
                data = data.to(device)
                # data = (data - data.min()) / (data.max() - data.min() + EPSILON)
                optimizer.zero_grad()
                recon_batch, mu, logvar = self(data)
                if batch_idx % 20 == 0:
                    sampled_mu.append(mu.detach())
                    sampled_logvar.append(logvar.detach())
                re_loss, kld_loss, loss = self.loss_function(
                    recon_batch,
                    data,
                    mu,
                    logvar,
                    epoch=epoch + 1,
                    normalize=True,
                    total_epochs=epochs + 1,
                    batch_idx=batch_idx,
                    n_batches_per_epoch=n_batches_per_epoch,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                reconstruct_loss += re_loss.item()
                kld_loss_total += kld_loss.item()
                if self.use_batch_cyclical:
                    current_beta = self.loss_balancer.get_beta(
                        epoch + 1, re_loss, kld_loss, batch_idx, n_batches_per_epoch
                    )
                    epoch_betas.append(current_beta)

            avg_loss = epoch_loss / len(train_loader)
            avg_re_loss = reconstruct_loss / len(train_loader)
            avg_kld_loss = kld_loss_total / len(train_loader)
            avg_beta = np.mean(epoch_betas) if epoch_betas else 0.0

            if sampled_mu:
                epoch_mu, epoch_logvar = torch.cat(sampled_mu, 0), torch.cat(
                    sampled_logvar, 0
                )
                kl_healthy = self.monitor_kl_health(epoch_mu, epoch_logvar)
                if not kl_healthy:
                    collapse_counter += 1

            if self.detect_posterior_collapse(train_loader, threshold=0.1):
                self.logger.warning(f"🚨 Collapse detected at epoch {epoch}")
                self.loss_balancer.min_beta = max(
                    0.001, self.loss_balancer.min_beta * 0.5
                )
                self.loss_balancer.max_beta = max(
                    0.1, self.loss_balancer.max_beta * 0.7
                )
                self.loss_balancer.gamma = max(0.001, self.loss_balancer.gamma * 0.5)
                collapse_counter += 1
                self.loss_balancer.recon_avg, self.loss_balancer.kld_avg = None, None

            history["epoch_losses"].append(avg_loss)
            history["reconstruct_losses"].append(avg_re_loss)
            history["kld_losses"].append(avg_kld_loss)
            history["beta_values"].append(avg_beta)
            # self.logger.info(
            #     f"Epoch {epoch}: Loss: {avg_loss:.6f}, RE: {avg_re_loss:.6f}, "
            #     f"KLD: {avg_kld_loss:.6f}, Avg Beta: {avg_beta:.4f}, "
            #     f"Collapse Counter: {collapse_counter}"
            # )
            self.logger.info(
                f"\n{'🚀 PRETRAIN EPOCH':<20} {epoch:3d} / {freeze_prior_epochs}\n"
                f"{'─'*60}\n"
                f"📊 Average Losses:\n"
                f"   ├─ Total:          {avg_loss:10.6f}\n"
                f"   ├─ Reconstruction: {avg_re_loss:10.6f}\n"
                f"   ├─ KL Divergence:  {avg_kld_loss:10.6f}\n"
                f"   └─ Beta (avg):     {avg_beta:10.6f}\n"
                f"🏥 Health Metrics:\n"
                f"   ├─ Collapse Count: {collapse_counter}\n"
                f"   └─ Status:         {'⚠️  Unstable' if collapse_counter > 2 else '✅ Stable'}\n"
                f"{'─'*60}\n"
            )

            if pretrain_checkpoint_path:
                torch.save(
                    {
                        "phase": "epochs",
                        "step_count": step_count,
                        "epoch": epoch + 1,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "history": history,
                        "collapse_counter": collapse_counter,
                        "loss_balancer_state": self.loss_balancer.get_state(),
                    },
                    pretrain_checkpoint_path,
                )

            if collapse_counter >= 5:
                self.logger.error("🚨 TOO MANY COLLAPSES DETECTED - STOPPING TRAINING")
                self.logger.error(
                    "Consider: lower beta, different architecture, or data preprocessing"
                )
                break

        if freeze_prior_epochs > 0:
            self.gradual_unfreeze(freeze_prior_epochs)

        k = round(len(train_loader) / 2)
        subset_size = min(k * self.batch_size, len(train_set))
        subset_indices = np.random.choice(
            len(train_set), size=subset_size, replace=False
        )
        subset_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_set, subset_indices),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        Z, Y = [], []
        self.eval()
        with torch.no_grad():
            for data, y in tqdm(subset_loader, desc="Extracting features"):
                data = data.to(device)
                # data = (data - data.min()) / (data.max() - data.min() + EPSILON)
                mu, _ = self.encode(data)
                Z.append(mu)
                Y.append(y)
        Z_cat = torch.cat(Z, 0).detach().cpu().numpy()
        Y_cat = torch.cat(Y, 0).detach().cpu().numpy()

        self.logger.info("Initializing clusters with K-means...")
        kmeans = KMeans(
            n_clusters=self.nClusters,
            random_state=42,
            n_init=20,
            max_iter=1000,  # More iterations
            algorithm="lloyd",
        )
        kmeans.fit(Z_cat)
        self.logger.info("Fitting GMM to latent space...")
        try:
            gmm = GaussianMixture(
                n_components=self.nClusters,
                covariance_type="diag",  # Keep diagonal
                max_iter=1000,  # More iterations (was 10000, but 1000 usually enough)
                means_init=kmeans.cluster_centers_,
                reg_covar=1e-4,  # Less regularization (was 1e-3)
                tol=1e-6,  # Tighter convergence
                n_init=5,  # Multiple random initializations
                init_params="random",  # Additional randomization
                verbose=1,  # See convergence info
                random_state=42,  # Fixed seed
            )
            gmm.fit_predict(Z_cat)
            if not gmm.converged_:
                self.logger.warning("GMM did not converge")
            if np.any(gmm.covariances_ <= 0):
                self.logger.warning("GMM produced invalid covariances")
            cluster_preds = gmm.fit_predict(Z_cat)
            self.pi_.data = torch.log(torch.tensor(gmm.weights_, device=device).float())
            self.mu_c.data = torch.tensor(gmm.means_, device=device).float()
            self.log_var_c.data = torch.log(
                torch.tensor(gmm.covariances_, device=device).float() + EPSILON
            )
            self.logger.info("Generating latent space visualization...")
            # try:
            #     self.visualize_latent_space(
            #         Z_cat, cluster_preds, Y_cat, context="pretrain_gmm"
            #     )
            # except Exception as e:
            #     self.logger.warning(
            #         f"Latent space visualization failed: {e}. Skipping..."
            #     )
        except Exception as e:
            self.logger.error(
                f"GMM fitting failed: {e}. Using K-means results instead."
            )
            cluster_preds = kmeans.labels_
            self.pi_.data = torch.log(
                torch.ones(self.nClusters, device=device).float() / self.nClusters
            )
            self.mu_c.data = torch.tensor(
                kmeans.cluster_centers_, device=device
            ).float()
            self.log_var_c.data = torch.full(
                (self.nClusters, self.latent_dim), -1.0, device=device
            )
            self.logger.info(
                "Generating latent space visualization (K-means fallback)..."
            )
        # self._evaluate_clustering(
        #     subset_loader, device, Y_cat, Z_cat, history, freeze_prior_epochs
        # )
        self.unfreeze_prior()
        self.train()
        return history

    def _evaluate_clustering(
        self,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        Y_cat: np.ndarray,
        Z_cat: np.ndarray,
        history: dict,
        current_epoch: int,
    ) -> None:
        self.eval()
        Z, Y, predictions = [], [], []
        with torch.no_grad():
            for data, y in tqdm(data_loader, desc="Evaluating clustering"):
                data = data.to(device)
                # data = (data - data.min()) / (data.max() - data.min() + EPSILON)
                batch_predictions = self.predict(data)
                predictions.append(batch_predictions)
                mu, _ = self.encode(data)
                Z.append(mu)
                Y.append(y)
        Z_cat = torch.cat(Z, 0).detach().cpu().numpy()
        Y_cat = torch.cat(Y, 0).detach().cpu().numpy()
        cluster_preds = np.concatenate(predictions, axis=0)
        if len(Y_cat) > 0:
            try:
                from sklearn.metrics import (
                    normalized_mutual_info_score,
                    adjusted_rand_score,
                    silhouette_score,
                    davies_bouldin_score,
                    calinski_harabasz_score,
                )

                Y_cat_1d = Y_cat.reshape(-1) if len(Y_cat.shape) > 1 else Y_cat
                nmi = normalized_mutual_info_score(Y_cat_1d, cluster_preds)
                ari = adjusted_rand_score(Y_cat_1d, cluster_preds)
                metrics = {}
                if len(np.unique(cluster_preds)) >= 2:
                    metrics["silhouette"] = silhouette_score(Z_cat, cluster_preds)
                    metrics["db_index"] = davies_bouldin_score(Z_cat, cluster_preds)
                    metrics["ch_index"] = calinski_harabasz_score(Z_cat, cluster_preds)
                else:
                    metrics["silhouette"], metrics["db_index"], metrics["ch_index"] = (
                        float("nan"),
                        float("nan"),
                        float("nan"),
                    )
                    self.logger.warning(
                        "Only one cluster found, clustering metrics cannot be computed"
                    )
                history["silhouette_scores"].append(metrics["silhouette"])
                history["db_scores"].append(metrics["db_index"])
                history["ch_scores"].append(metrics["ch_index"])
                history["nmi_scores"].append(nmi)
                history["ari_scores"].append(ari)
                silhouette_val = (
                    metrics["silhouette"]
                    if not np.isnan(metrics["silhouette"])
                    else float("nan")
                )

                self.logger.info(
                    f"Epoch {current_epoch} - nmi: {nmi:.6f}, ari: {ari:.6f}, "
                    f"silhouette: {silhouette_val:.6f}"
                    f"db: {metrics['db_index']:.6f if not np.isnan(metrics['db_index']) else 'N/A'}, "
                    f"ch: {metrics['ch_index']:.6f if not np.isnan(metrics['ch_index']) else 'N/A'}"
                )
            except Exception as e:
                self.logger.error(f"Error computing clustering metrics: {e}")
                history["silhouette_scores"].append(float("nan"))
                history["db_scores"].append(float("nan"))
                history["ch_scores"].append(float("nan"))
                history["nmi_scores"].append(float("nan"))
                history["ari_scores"].append(float("nan"))
        try:
            self.visualize_latent_space(
                Z_cat, cluster_preds, Y_cat_1d, context="evaluation"
            )
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
        self.train()

    def visualize_latent_space(
        self,
        Z: np.ndarray,
        pred_labels: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        context: str = "general",
    ) -> None:
        if np.isnan(Z).any():
            self.logger.warning(
                f"Found {np.isnan(Z).sum()} NaN values in latent space."
            )
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="mean")
            Z_clean = imputer.fit_transform(Z)
            self.logger.info("Imputed NaN values with mean values.")
        else:
            Z_clean = Z
        try:
            self.logger.info("Applying t-SNE dimensionality reduction...")
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
            plt.title("Latent Space Visualization (t-SNE)", fontsize=14)
            plt.xlabel("t-SNE Dimension 1", fontsize=12)
            plt.ylabel("t-SNE Dimension 2", fontsize=12)
            if true_labels is not None:
                plt.legend(loc="upper right")
            plt.tight_layout()
            output_dir = os.path.join("images", "clustering")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir, f"cluster_latent_space_{context}_k{self.nClusters}.png"
            )
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Visualization saved to {output_path}")
            plt.close()
        except Exception as e:
            self.logger.error(f"Error during visualization: {e}")


def plot_training_history(
    history: dict, output_dir: str = "images/clustering", logger=None
) -> None:
    if logger is None:
        logger = logging.getLogger("vae_clustering")
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.plot(history["epoch_losses"], label="Total Loss", color="red")
    plt.xlabel("Step/Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Total Loss")
    plt.yscale("log")
    plt.subplot(2, 3, 2)
    plt.plot(history["reconstruct_losses"], label="Reconstruction Loss", color="blue")
    plt.xlabel("Step/Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Reconstruction Loss")
    plt.yscale("log")
    plt.subplot(2, 3, 3)
    plt.plot(history["kld_losses"], label="KLD Loss", color="green")
    plt.xlabel("Step/Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("KLD Loss")
    plt.yscale("log")
    if "beta_values" in history and len(history["beta_values"]) > 0:
        plt.subplot(2, 3, 4)
        plt.plot(history["beta_values"], label="Beta Values", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Beta")
        plt.legend()
        plt.title("Beta Annealing Schedule")
    if "nmi_scores" in history and len(history["nmi_scores"]) > 0:
        plt.subplot(2, 3, 5)
        plt.plot(history["nmi_scores"], label="NMI", color="purple")
        plt.plot(history["ari_scores"], label="ARI", color="brown")
        if "v_measure_scores" in history and history["v_measure_scores"]:
            plt.plot(
                history["v_measure_scores"],
                label="V-Measure",
                color="orange",
                linestyle=":",
            )

        plt.xlabel("Evaluation Point")
        plt.ylabel("Score")
        plt.legend()
        plt.title("Clustering Performance")
        plt.subplot(2, 3, 6)
        valid_silhouette = [x for x in history["silhouette_scores"] if not np.isnan(x)]
        valid_db = [x for x in history["db_scores"] if not np.isnan(x)]
        valid_ch = [x for x in history["ch_scores"] if not np.isnan(x)]
        if valid_silhouette:
            plt.plot(
                range(len(valid_silhouette)),
                valid_silhouette,
                label="Silhouette",
                color="cyan",
            )
        if valid_db:
            plt.plot(
                range(len(valid_db)), valid_db, label="Davies-Bouldin", color="magenta"
            )
        if valid_ch:
            plt.plot(
                range(len(valid_ch)),
                valid_ch,
                label="Calinski-Harabasz",
                color="yellow",
                linestyle="--",
            )
        plt.xlabel("Evaluation Point")
        plt.ylabel("Score")
        plt.legend()
        plt.title("Internal Clustering Metrics")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "comprehensive_training_history.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    logger.info(f"Comprehensive training history plots saved to {output_dir}")


def create_model_with_batch_cyclical(
    latent_dim: int = 10,
    nClusters: int = 8,
    batch_size: int = 128,
    logger=None,
    device=None,
    n_cycles_per_epoch: int = 5,
    cycle_ratio: float = 0.5,
    gamma: float = 0.01,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger is None:
        import logging

        logger = logging.getLogger("vae_clustering")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    model = MyModel(
        latent_dim=latent_dim,
        nClusters=nClusters,
        batch_size=batch_size,
        logger=logger,
        device=device,
        use_cyclical_annealing=False,
        use_batch_cyclical=True,
        nc=1,
        ndf=64,
        ngf=64,
        # ndf=32,
        # ngf=32,
    )
    model.loss_balancer.n_cycles_per_epoch = n_cycles_per_epoch
    model.loss_balancer.cycle_ratio = cycle_ratio
    model.loss_balancer.gamma = gamma
    logger.info("Created VAE model with batch-level cyclical annealing:")
    logger.info(f"  - Latent dimensions: {latent_dim}")
    logger.info(f"  - Number of clusters: {nClusters}")
    logger.info(f"  - Cycles per epoch: {n_cycles_per_epoch}")
    logger.info(f"  - Cycle ratio: {cycle_ratio}")
    logger.info(f"  - Gamma: {gamma}")
    logger.info(
        f"  - Beta range: [{model.loss_balancer.min_beta}, {model.loss_balancer.max_beta}]"
    )
    return model


def train_with_monitoring(
    model: MyModel,
    train_loader: torch.utils.data.DataLoader,
    train_set: torch.utils.data.Dataset,
    learning_rate: float = 1e-3,
    pretrain_epochs: int = 5,
    gamma_steps: int = 1000,
    freeze_prior_epochs: int = 0,
    device=None,
):
    if device is None:
        device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.logger.info("=" * 80)
    model.logger.info("STARTING TRAINING WITH BATCH-LEVEL CYCLICAL ANNEALING")
    model.logger.info("=" * 80)
    model.logger.info("Training configuration:")
    model.logger.info(f"  - Learning rate: {learning_rate}")
    model.logger.info(f"  - Pretrain epochs: {pretrain_epochs}")
    model.logger.info(f"  - Gamma steps: {gamma_steps}")
    model.logger.info(f"  - Freeze prior epochs: {freeze_prior_epochs}")
    model.logger.info(f"  - Batches per epoch: {len(train_loader)}")
    model.logger.info(f"  - Use batch cyclical: {model.use_batch_cyclical}")
    history = model.pretrain(
        train_loader=train_loader,
        optimizer=optimizer,
        train_set=train_set,
        epochs=pretrain_epochs,
        gamma_steps=gamma_steps,
        freeze_prior_epochs=freeze_prior_epochs,
        device=device,
    )
    model.logger.info("=" * 80)
    model.logger.info("PRETRAINING COMPLETED")
    model.logger.info("=" * 80)
    model.check_prior_health()
    latent_healthy = model.check_latent_space_health(train_loader)
    collapse_detected = model.detect_posterior_collapse(train_loader, threshold=0.1)
    if collapse_detected:
        model.logger.error("🚨 POSTERIOR COLLAPSE DETECTED AFTER TRAINING!")
        model.logger.error(
            "Training may need adjustment of beta parameters or architecture."
        )
    elif latent_healthy:
        model.logger.info(
            "✅ Training completed successfully - latent space appears healthy"
        )
    return history


if __name__ == "__main__":
    import torch.utils.data as data_utils

    dummy_data = torch.randn(1000, 1, 64, 64)
    dummy_labels = torch.randint(0, 8, (1000,))
    dataset = data_utils.TensorDataset(dummy_data, dummy_labels)
    train_loader = data_utils.DataLoader(dataset, batch_size=128, shuffle=True)
    model = create_model_with_batch_cyclical(
        latent_dim=10,
        nClusters=8,
        batch_size=128,
        n_cycles_per_epoch=5,
        cycle_ratio=0.5,
        gamma=0.01,
    )
    history = train_with_monitoring(
        model=model,
        train_loader=train_loader,
        train_set=dataset,
        learning_rate=1e-3,
        pretrain_epochs=5,
        gamma_steps=500,
        freeze_prior_epochs=0,
    )
    plot_training_history(history)
    print("Training completed! Check the generated plots and logs for results.")
