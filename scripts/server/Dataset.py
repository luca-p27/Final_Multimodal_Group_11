"""
Dataset.py

CrypticBioDataset gives back (img, geo, label). Images are tried from local_path first,
then downloaded from url and cached in memory. Samples with no image are dropped.

geo is float32 (geo_dim,) for continuous encoders or int64 (2,) for discrete ones.
Pass top_n to load_dataframe() to keep only the N most common species.
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

try:
    import requests
    from io import BytesIO
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

from Encoder import (
    CONTINUOUS_ENCODERS,
    latlon_to_hex_idx,
    label_to_idx,
)


def clean_metadata(df: pd.DataFrame, url_map: dict = None) -> pd.DataFrame:
    """
    Drop rows with missing or invalid core fields.
    """
    required = ['scientificName', 'decimalLatitude', 'decimalLongitude']
    df = df.dropna(subset=[c for c in required if c in df.columns])
    if 'scientificName' in df.columns:
        df = df[df['scientificName'].str.strip() != '']
    if 'decimalLatitude' in df.columns and 'decimalLongitude' in df.columns:
        df = df[
            df['decimalLatitude'].between(-90,  90) &
            df['decimalLongitude'].between(-180, 180)
        ]
    if 'url' in df.columns:
        df = df.dropna(subset=['url'])
        df = df[df['url'].str.strip() != '']

    if url_map and 'url' in df.columns:
        before = len(df)
        df = df[df['url'].map(lambda u: os.path.exists(url_map.get(u, '')))]
        dropped = before - len(df)
        if dropped:
            warnings.warn(f"Dropped {dropped} rows: local image file not found.")

    return df.reset_index(drop=True)


def load_dataframe(path: str, top_n: int = None, min_samples: int = 5,
                   url_map: dict = None) -> pd.DataFrame:
    """
    Load a TSV or CSV, clean it, normalise the species column, and optionally
    restrict to the top-N most common species.

    Args:
        path        : path to the data file (.tsv or .csv)
        top_n       : keep only the N most common species (None = keep all)
        min_samples : drop species with fewer than this many samples
        url_map     : optional dict {url: local_path} — rows whose image file
                      does not exist on disk are dropped before training

    Returns:
        cleaned DataFrame with a 'species' column
    """
    sep = '\t' if path.endswith('.tsv') else ','
    df  = pd.read_csv(path, sep=sep)
    df  = clean_metadata(df, url_map=url_map)

    if 'species' not in df.columns:
        df['species'] = df['scientificName']

    counts = df['species'].value_counts()
    df = df[df['species'].isin(counts[counts >= min_samples].index)]

    if top_n and top_n > 0:
        top_names = counts.nlargest(top_n).index
        df = df[df['species'].isin(top_names)]

    return df.reset_index(drop=True)


class CrypticBioDataset(Dataset):
    """
    Unified dataset for all encoder types.

    Args:
        df              : DataFrame (must have scientificName, decimalLatitude,
                          decimalLongitude; optionally url, local_path,
                          country, continent)
        encoder_type    : 'wrap', 'raw', 'sh', 'hex', 'geo_label' or 'None'
        transform       : torchvision transform applied to each image
        label_encoder   : fitted sklearn LabelEncoder; if None, fit on this df
        image_cache     : shared dict {url: PIL.Image} — pass the same object
                          to train/val/test datasets to avoid re-downloading
        geo_encoder_obj : instantiated WrapEncoder / RawEncoder / SHEncoder
                          (required when encoder_type is continuous)
        vocab1          : coarse_hex_vocab or country_vocab
                          (required when encoder_type is discrete)
        vocab2          : fine_hex_vocab or continent_vocab
                          (required when encoder_type is discrete)
        h3_coarse       : H3 coarse resolution  (hex encoder only, default 4)
        h3_fine         : H3 fine resolution    (hex encoder only, default 6)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        encoder_type: str,
        transform=None,
        label_encoder: LabelEncoder = None,
        image_cache: dict = None,
        geo_encoder_obj=None,
        vocab1: list = None,
        vocab2: list = None,
        h3_coarse: int = 4,
        h3_fine:   int = 6,
    ):
        self.df           = df.reset_index(drop=True)
        self.encoder_type = encoder_type
        self.transform    = transform
        self.image_cache  = image_cache if image_cache is not None else {}
        if 'species' not in self.df.columns:
            self.df = self.df.copy()
            self.df['species'] = self.df['scientificName']

        if label_encoder is not None:
            self.label_encoder = label_encoder
            self.labels = label_encoder.transform(self.df['species'])
        else:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(self.df['species'])

        lats = self.df['decimalLatitude'].values.astype(float)
        lons = self.df['decimalLongitude'].values.astype(float)

        if encoder_type in CONTINUOUS_ENCODERS:
            if geo_encoder_obj is None:
                raise ValueError("geo_encoder_obj required for continuous encoders")
            self.geo_data = geo_encoder_obj.encode_batch(lats, lons)

        elif encoder_type == 'hex':
            if vocab1 is None or vocab2 is None:
                raise ValueError("vocab1 and vocab2 (coarse/fine) required for hex encoder")
            indices = [
                (latlon_to_hex_idx(lat, lon, h3_coarse, vocab1),
                 latlon_to_hex_idx(lat, lon, h3_fine,   vocab2))
                for lat, lon in zip(lats, lons)
            ]
            self.geo_data = torch.tensor(indices, dtype=torch.long)

        elif encoder_type == 'geo_label':
            if vocab1 is None or vocab2 is None:
                raise ValueError("vocab1 (country) and vocab2 (continent) required for geo_label encoder")
            indices = [
                (label_to_idx(row.get('country',   ''), vocab1),
                 label_to_idx(row.get('continent', ''), vocab2))
                for _, row in self.df.iterrows()
            ]
            self.geo_data = torch.tensor(indices, dtype=torch.long)
        elif encoder_type == "None":
            indices = [
                (label_to_idx(row.get('country',   ''), vocab1),
                 label_to_idx(row.get('continent', ''), vocab2))
                for _, row in self.df.iterrows()
            ]
            self.geo_data = torch.tensor(indices, dtype=torch.long)
        else:
            raise ValueError(f"Unknown encoder_type '{encoder_type}'")

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, row: pd.Series):
        # try local path first
        lp = row.get('local_path', None)
        if lp is not None and not pd.isna(lp) and os.path.exists(str(lp)):
            try:
                return Image.open(str(lp)).convert('RGB')
            except Exception:
                pass

        # fall back to URL download, cached
        url = str(row.get('url', ''))
        if url in self.image_cache:
            return self.image_cache[url]
        if _HAS_REQUESTS and url.startswith('http'):
            try:
                resp = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert('RGB')
                self.image_cache[url] = img
                return img
            except Exception:
                pass
        return None  # caller will skip this sample

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_image(row)
        if img is None:
            return None
        if self.transform:
            img = self.transform(img)
        geo   = self.geo_data[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, geo, label


def collate_skip_none(batch):
    """DataLoader collate_fn that silently drops samples where image loading failed."""
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)