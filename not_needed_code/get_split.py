"""
get_split.py

Splits the dataset and saves the train/test CSVs to disk.

Usage:
    python get_split.py --data_path input/CrypticBio-Common_continent.tsv
"""

import argparse
import json
import os
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms

from scripts.server.Encoder import (
    ALL_ENCODER_NAMES,
    CONTINUOUS_ENCODERS, DISCRETE_ENCODERS,
    HexGridEncoder, GeoLabelEncoder,
    build_continuous_encoder,
    build_hex_vocab, build_label_vocab,
)
from scripts.server.Model   import build_model
from scripts.server.Dataset import CrypticBioDataset, load_dataframe, collate_skip_none
from scripts.server.Train   import fit, evaluate

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
DATA_DEFAULT  = os.path.join(_PROJECT_ROOT, 'input', 'CrypticBio-Common_continent.tsv')
OUT_DEFAULT   = os.path.join(_PROJECT_ROOT, 'results', 'encoders')

TRAIN_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def parse_args():
    p = argparse.ArgumentParser(
        description='Merged biodiversity multimodal classification pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data_path', default=DATA_DEFAULT,
                   help='path to TSV or CSV data file')
    p.add_argument('--top_n', type=int, default=0,
                   help='keep only top-N species (0 = use all species)')
    p.add_argument('--encoding', default='hex',
                   help=f'encoder(s): {ALL_ENCODER_NAMES} or "all" '
                        f'(comma-separated for multiple, e.g. "wrap,hex")')
    p.add_argument('--fusion', default='both',
                   choices=['early', 'late', 'both'],
                   help='fusion strategy — applies to all encoder types')
    p.add_argument('--out_dir', default=OUT_DEFAULT,
                   help='directory where models and results are written')
    p.add_argument('--epochs', type=int, default=3,
                   help='maximum training epochs per run')
    p.add_argument('--patience', type=int, default=5,
                   help='early-stopping patience (epochs without val improvement)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--emb_dim', type=int, default=64,
                   help='embedding dim for HexGridEncoder / GeoLabelEncoder')
    p.add_argument('--h3_coarse', type=int, default=4,
                   help='H3 coarse resolution (hex encoder)')
    p.add_argument('--h3_fine', type=int, default=6,
                   help='H3 fine resolution (hex encoder)')
    p.add_argument('--geo_mode', default='both',
                   choices=['country', 'continent', 'both'],
                   help='which label dimensions to embed (geo_label encoder)')
    p.add_argument('--url_map', default=None,
                   help='path to url_to_path.csv from download_images.py; '
                        'auto-detected from common server/SSD locations if not set')
    return p.parse_args()


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_encodings(arg: str) -> list:
    if arg.lower() == 'all':
        return list(ALL_ENCODER_NAMES)
    return [e.strip() for e in arg.split(',')]


def resolve_fusion_types(arg: str) -> list:
    return ['early', 'late'] if arg == 'both' else [arg]


def split_df(df: pd.DataFrame, seed: int):
    """Stratified 70/15/15 split with safe fallback when classes are few."""
    train, temp = train_test_split(
        df, test_size=0.3, random_state=seed,
        stratify=df['species'])
    val, test = train_test_split(
        temp, test_size=0.5, random_state=seed,
        stratify=temp['species'])
    return train, val, test


def make_loaders(train_df, val_df, test_df, encoder_type, label_enc,
                 batch_size, num_workers, image_cache, **ds_kwargs):
    """Build train/val/test DataLoaders for any encoder type."""
    pin = torch.cuda.is_available()
    def mk(df, tf):
        return CrypticBioDataset(df, encoder_type,
                                 transform=tf,
                                 label_encoder=label_enc,
                                 image_cache=image_cache,
                                 **ds_kwargs)
    return (
        DataLoader(mk(train_df, TRAIN_TF), batch_size,
                   shuffle=True,  num_workers=num_workers, pin_memory=pin,
                   collate_fn=collate_skip_none),
        DataLoader(mk(val_df,   EVAL_TF),  batch_size,
                   shuffle=False, num_workers=num_workers, pin_memory=pin,
                   collate_fn=collate_skip_none),
        DataLoader(mk(test_df,  EVAL_TF),  batch_size,
                   shuffle=False, num_workers=num_workers, pin_memory=pin,
                   collate_fn=collate_skip_none),
    )


def save_results(out_dir, tag, label_enc, all_labels, all_preds,
                 all_probs, test_acc, extra: dict) -> dict:
    """Write predictions, per-class metrics, confusion matrix, and JSON summary."""
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame({
        'True_Species':      label_enc.inverse_transform(all_labels),
        'Predicted_Species': label_enc.inverse_transform(all_preds),
        'Confidence':        [max(p) for p in all_probs],
    }).to_csv(os.path.join(out_dir, f'test_predictions_{tag}.csv'), index=False)

    present = sorted(set(all_labels) | set(all_preds))
    pd.DataFrame(
        classification_report(all_labels, all_preds,
                               labels=present,
                               target_names=label_enc.classes_[present],
                               output_dict=True)
    ).transpose().to_csv(os.path.join(out_dir, f'per_class_metrics_{tag}.csv'))

    np.savetxt(os.path.join(out_dir, f'confusion_matrix_{tag}.csv'),
               confusion_matrix(all_labels, all_preds, labels=present),
               delimiter=',', fmt='%d')

    metrics = {
        **extra,
        'test_accuracy':   round(float(test_acc / 100), 4),
        'macro_f1':        round(float(f1_score(all_labels, all_preds,
                                                average='macro',    zero_division=0)), 4),
        'weighted_f1':     round(float(f1_score(all_labels, all_preds,
                                                average='weighted', zero_division=0)), 4),
        'macro_precision': round(float(precision_score(all_labels, all_preds,
                                                       average='macro', zero_division=0)), 4),
        'macro_recall':    round(float(recall_score(all_labels, all_preds,
                                                    average='macro', zero_division=0)), 4),
    }
    with open(os.path.join(out_dir, f'metrics_{tag}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


def run_experiment(encoder_type, fusion_type,
                   train_df, val_df, test_df, label_enc,
                   args, device, image_cache) -> dict:
    """
    Train and evaluate one encoder x fusion combination.

    Returns a metrics dict that is appended to the summary table.
    """
    num_classes = len(label_enc.classes_)
    tag         = f"{encoder_type}_{fusion_type}"
    model_path  = os.path.join(args.out_dir, f'best_model_{tag}.pth')

    print(f"\n{'='*58}")
    print(f"  Encoder: {encoder_type.upper():10s}  Fusion: {fusion_type.upper()}")
    print(f"{'='*58}")

    ds_kwargs = {}

    if encoder_type in CONTINUOUS_ENCODERS:
        enc_obj = build_continuous_encoder(encoder_type)
        ds_kwargs['geo_encoder_obj'] = enc_obj
        geo_dim = enc_obj.out_dim
        geo_encoder_module = None
        print(f"  geo_dim: {geo_dim}")

    elif encoder_type == 'hex':
        coarse_vocab = build_hex_vocab(train_df, args.h3_coarse)
        fine_vocab   = build_hex_vocab(train_df, args.h3_fine)
        geo_encoder_module = HexGridEncoder(
            len(coarse_vocab), len(fine_vocab), args.emb_dim)
        ds_kwargs.update(vocab1=coarse_vocab, vocab2=fine_vocab,
                         h3_coarse=args.h3_coarse, h3_fine=args.h3_fine)
        geo_dim = None
        print(f"  H3 resolutions : coarse={args.h3_coarse}  fine={args.h3_fine}")
        print(f"  Vocab sizes    : coarse={len(coarse_vocab)}  fine={len(fine_vocab)}")

    elif encoder_type == 'geo_label':
        if 'country' not in train_df.columns or 'continent' not in train_df.columns:
            print("  [skip] 'country'/'continent' columns not found in data.")
            return {}
        country_vocab   = build_label_vocab(train_df, 'country')
        continent_vocab = build_label_vocab(train_df, 'continent')
        geo_encoder_module = GeoLabelEncoder(
            len(country_vocab), len(continent_vocab),
            emb_dim=args.emb_dim, mode=args.geo_mode)
        ds_kwargs.update(vocab1=country_vocab, vocab2=continent_vocab)
        geo_dim = None
        print(f"  geo_mode       : {args.geo_mode}")
        print(f"  Country vocab  : {len(country_vocab)}")
        print(f"  Continent vocab: {len(continent_vocab)}")

    is_cuda     = torch.cuda.is_available()
    batch_size  = 32 if is_cuda else 16
    num_workers = 0

    train_loader, val_loader, test_loader = make_loaders(
        train_df, val_df, test_df, encoder_type, label_enc,
        batch_size, num_workers, image_cache, **ds_kwargs)

    model = build_model(encoder_type, fusion_type, num_classes,
                        geo_dim=geo_dim, geo_encoder=geo_encoder_module)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    base  = model.module if isinstance(model, nn.DataParallel) else model

    optimizer = optim.AdamW(
        base.param_groups(backbone_lr=3e-5, head_lr=1e-3),
        weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2)

    os.makedirs(args.out_dir, exist_ok=True)
    fit(model, train_loader, val_loader, optimizer, scheduler,
        args.epochs, device, model_path, args.patience)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    _, test_acc, all_preds, all_labels, all_probs = evaluate(
        model, test_loader, criterion, device)

    metrics = save_results(
        args.out_dir, tag, label_enc,
        all_labels, all_preds, all_probs,
        test_acc,
        {'encoder': encoder_type, 'fusion': fusion_type, 'num_classes': num_classes},
    )
    print(f"\n  >> test_acc={test_acc:.2f}%  macro_f1={metrics['macro_f1']:.4f}")
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Device  : {device}")
    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"Data    : {args.data_path}")
    print(f"Out dir : {args.out_dir}")

    if not os.path.exists(args.data_path):
        print(f"\nError: data file not found: {args.data_path}")
        return

    # auto-detect url_to_path.csv from common locations if not explicitly set
    url_map = {}
    candidates = [
        args.url_map,
        '/data/s4610601/MultiModal_images/url_to_path.csv',
        '/Volumes/PRO-G40/MultiModal_images/url_to_path.csv',
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            um = pd.read_csv(candidate)
            url_map = dict(zip(um['url'], um['local_path']))
            print(f"url_map : {len(url_map):,} local paths loaded from {candidate}")
            break
    if not url_map:
        print("url_map : not found — images will be downloaded on-the-fly")

    df = load_dataframe(args.data_path, top_n=args.top_n, url_map=url_map or None)
    print(f"\nLoaded  : {len(df):,} rows  |  {df['species'].nunique()} species"
          + (f"  (top {args.top_n})" if args.top_n else ""))

    if url_map and 'url' in df.columns:
        df['local_path'] = df['url'].map(url_map)

    train_df, val_df, test_df = split_df(df, args.seed)
    print(f"Split   : train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    name = args.data_path.split('.')[0]
    train_df.to_csv(f"{name}_trainsplit.csv", sep=',')
    test_df.to_csv(f"{name}_testsplit.csv", sep=',')





if __name__ == '__main__':
    main()