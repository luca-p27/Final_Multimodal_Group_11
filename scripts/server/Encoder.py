"""
Encoder.py

Continuous encoders (WrapEncoder, RawEncoder, SHEncoder) convert lat/lon to fixed
vectors at dataset init time. Discrete encoders (HexGridEncoder, GeoLabelEncoder)
are nn.Modules that learn embeddings from integer indices precomputed in Dataset.
"""

import numpy as np
import torch
import torch.nn as nn
import h3
from scipy.special import sph_harm

_H3_VERSION = int(h3.__version__.split('.')[0])

def _latlng_to_cell(lat: float, lon: float, res: int) -> str:
    return h3.latlng_to_cell(lat, lon, res) if _H3_VERSION >= 4 else h3.geo_to_h3(lat, lon, res)


# Continuous encoders

class WrapEncoder:
    """Sinusoidal encoding (Mac Aodha et al., 2019). out_dim = 4."""
    out_dim = 4

    def encode_batch(self, lats: np.ndarray, lons: np.ndarray) -> torch.Tensor:
        lat_r = np.pi * lats / 90.0
        lon_r = np.pi * lons / 180.0
        enc = np.stack([np.sin(lat_r), np.cos(lat_r),
                        np.sin(lon_r), np.cos(lon_r)], axis=1)
        return torch.tensor(enc, dtype=torch.float32)


class RawEncoder:
    """Raw normalised lat/lon to [-1, 1]. out_dim = 2."""
    out_dim = 2

    def encode_batch(self, lats: np.ndarray, lons: np.ndarray) -> torch.Tensor:
        enc = np.stack([lats / 90.0, lons / 180.0], axis=1)
        return torch.tensor(enc, dtype=torch.float32)


class SHEncoder:
    """Real spherical harmonics (Russwurm et al., 2024). out_dim = 121."""
    MAX_DEGREE = 10
    out_dim = (MAX_DEGREE + 1) ** 2  # 121

    def encode_batch(self, lats: np.ndarray, lons: np.ndarray) -> torch.Tensor:
        theta = np.radians(lons)               # azimuthal angle
        phi   = np.pi / 2 - np.radians(lats)  # colatitude
        features = []
        for l in range(self.MAX_DEGREE + 1):
            for m in range(-l, l + 1):
                if m < 0:
                    sh = sph_harm(abs(m), l, theta, phi)
                    features.append(np.sqrt(2) * sh.imag)
                elif m == 0:
                    sh = sph_harm(0, l, theta, phi)
                    features.append(sh.real)
                else:
                    sh = sph_harm(m, l, theta, phi)
                    features.append(np.sqrt(2) * sh.real)
        enc = np.stack(features, axis=1).astype(np.float32)
        return torch.tensor(enc, dtype=torch.float32)


# Vocabulary helpers for discrete encoders

def build_hex_vocab(df, resolution: int) -> list:
    """
    Build a sorted H3 cell vocabulary from a DataFrame with
    decimalLatitude / decimalLongitude columns. Index 0 = <UNK>.
    """
    cells = set()
    for _, row in df.iterrows():
        cells.add(_latlng_to_cell(
            float(row['decimalLatitude']),
            float(row['decimalLongitude']),
            resolution,
        ))
    return ['<UNK>'] + sorted(cells)


def latlon_to_hex_idx(lat: float, lon: float, resolution: int, vocab: list) -> int:
    """Map (lat, lon) to a vocab index; returns 0 (<UNK>) if unseen."""
    cell = _latlng_to_cell(lat, lon, resolution)
    try:
        return vocab.index(cell)
    except ValueError:
        return 0


def build_label_vocab(df, col: str) -> list:
    """
    Build a sorted vocabulary for a categorical column (e.g. 'country',
    'continent'). Index 0 = <UNK>.
    """
    return ['<UNK>'] + sorted(df[col].dropna().astype(str).unique())


def label_to_idx(value, vocab: list) -> int:
    """Map a category string to its vocab index; returns 0 (<UNK>) if absent."""
    try:
        return vocab.index(str(value))
    except ValueError:
        return 0


# Discrete encoder nn.Modules

class HexGridEncoder(nn.Module):
    """
    Learnable H3 hexagonal grid encoder (two resolution levels).

    Inputs : idx1 (coarse_idx), idx2 (fine_idx) — LongTensors (B,)
    Output : float32 Tensor (B, 2 * emb_dim)
    """

    def __init__(self, coarse_vocab_size: int, fine_vocab_size: int, emb_dim: int = 64):
        super().__init__()
        self.emb_dim = emb_dim
        self.out_dim = 2 * emb_dim
        self.coarse_emb = nn.Embedding(coarse_vocab_size, emb_dim, padding_idx=0)
        self.fine_emb   = nn.Embedding(fine_vocab_size,   emb_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(2 * emb_dim, 2 * emb_dim),
            nn.LayerNorm(2 * emb_dim),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.coarse_emb.weight, std=0.01)
        nn.init.normal_(self.fine_emb.weight,   std=0.01)
        with torch.no_grad():
            self.coarse_emb.weight[0].zero_()
            self.fine_emb.weight[0].zero_()

    def forward(self, idx1: torch.Tensor, idx2: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.coarse_emb(idx1), self.fine_emb(idx2)], dim=1)
        return self.proj(feat)

    def param_groups(self, lr: float = 1e-3) -> list:
        return [
            {'params': self.coarse_emb.parameters(), 'lr': lr},
            {'params': self.fine_emb.parameters(),   'lr': lr},
            {'params': self.proj.parameters(),        'lr': lr},
        ]


class GeoLabelEncoder(nn.Module):
    """
    Learnable categorical encoder for country / continent labels.

    Inputs : idx1 (country_idx), idx2 (continent_idx) — LongTensors (B,)
    Output : float32 Tensor (B, out_dim)
      out_dim = emb_dim      when mode in ('country', 'continent')
      out_dim = 2 * emb_dim  when mode == 'both'

    mode: 'country', 'continent', or 'both'
    """
    MODES = ('country', 'continent', 'both')

    def __init__(self, country_vocab_size: int, continent_vocab_size: int,
                 emb_dim: int = 64, mode: str = 'both'):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got '{mode}'")
        self.mode    = mode
        self.emb_dim = emb_dim
        concat_dim   = emb_dim * (2 if mode == 'both' else 1)
        self.out_dim = concat_dim

        if mode in ('country', 'both'):
            self.country_emb   = nn.Embedding(country_vocab_size,   emb_dim, padding_idx=0)
        if mode in ('continent', 'both'):
            self.continent_emb = nn.Embedding(continent_vocab_size, emb_dim, padding_idx=0)

        self.proj = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.LayerNorm(concat_dim),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for attr in ('country_emb', 'continent_emb'):
            if hasattr(self, attr):
                emb = getattr(self, attr)
                nn.init.normal_(emb.weight, std=0.01)
                with torch.no_grad():
                    emb.weight[0].zero_()

    def forward(self, idx1: torch.Tensor, idx2: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.mode in ('country', 'both'):
            parts.append(self.country_emb(idx1))
        if self.mode in ('continent', 'both'):
            parts.append(self.continent_emb(idx2))
        return self.proj(torch.cat(parts, dim=1))

    def param_groups(self, lr: float = 1e-3) -> list:
        groups = [{'params': self.proj.parameters(), 'lr': lr}]
        for attr in ('country_emb', 'continent_emb'):
            if hasattr(self, attr):
                groups.append({'params': getattr(self, attr).parameters(), 'lr': lr})
        return groups


# Encoder registry

CONTINUOUS_ENCODERS = {'wrap', 'raw', 'sh'}
DISCRETE_ENCODERS   = {'hex', 'geo_label'}
ALL_ENCODER_NAMES   = sorted(CONTINUOUS_ENCODERS | DISCRETE_ENCODERS)


def build_continuous_encoder(name: str):
    """Instantiate a continuous encoder by name."""
    mapping = {'wrap': WrapEncoder, 'raw': RawEncoder, 'sh': SHEncoder}
    if name not in mapping:
        raise ValueError(f"Unknown continuous encoder '{name}'. "
                         f"Available: {list(mapping)}")
    return mapping[name]()