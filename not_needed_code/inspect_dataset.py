"""
inspect_dataset.py

Sanity-check script for the dataset before training. Prints species distribution,
geographic column availability, split quality, stratification deviation, and image
availability via url_map. Saves the three splits to split_inspection/.
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scripts.server.Dataset import load_dataframe

def parse_args():
    p = argparse.ArgumentParser(description='Inspect dataset splits')
    p.add_argument('--data_path', default="/data/Multimodal/MultiModal_Assignment/input/CrypticBio-Common_continent_local.tsv",
                   help='path to TSV or CSV data file')
    p.add_argument('--top_n', type=int, default=0,
                   help='keep only top-N species (0 = use all)')
    p.add_argument('--seed', type=int, default=42,
                   help='random seed for reproducible splits')
    return p.parse_args()

def split_df(df: pd.DataFrame, seed: int):
    train, temp = train_test_split(
        df, test_size=0.3, random_state=seed,
        stratify=df['species'])
    val, test = train_test_split(
        temp, test_size=0.5, random_state=seed,
        stratify=temp['species'])
    return train, val, test

def main():
    args = parse_args()
    
    print("DATASET INSPECTION")
    print(f"Data path: {args.data_path}")
    print(f"Top N species: {args.top_n if args.top_n else 'ALL'}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Load data (same as Main.py)
    df = load_dataframe(args.data_path, top_n=args.top_n)
    print(f"Total samples loaded: {len(df):,}")
    print(f"Total species: {df['species'].nunique():,}")
    
    # Show species distribution
    print("SPECIES DISTRIBUTION")
    species_counts = df['species'].value_counts()
    for i, (species, count) in enumerate(species_counts.items()):
        pct = 100 * count / len(df)
        print(f"{i+1:2d}. {species[:45]:45s} {count:5d} samples ({pct:.1f}%)")
    
    # Show geo columns available
    print("GEOGRAPHIC COLUMNS")
    geo_cols = ['country', 'continent', 'decimalLatitude', 'decimalLongitude']
    for col in geo_cols:
        if col in df.columns:
            nunique = df[col].nunique() if col in ['country', 'continent'] else 'N/A'
            print(f" {col}: present, {nunique} unique values")
        else:
            print(f" {col}: NOT present")
    
    # Create splits (same as Main.py)

    print("CREATING SPLITS (70/15/15 stratified by species)")
    train_df, val_df, test_df = split_df(df, args.seed)
    
    print(f"Train: {len(train_df):,} samples ({100*len(train_df)/len(df):.1f}%)")
    print(f"Val:   {len(val_df):,} samples ({100*len(val_df)/len(df):.1f}%)")
    print(f"Test:  {len(test_df):,} samples ({100*len(test_df)/len(df):.1f}%)")
    
    # Check stratification quality
    print("STRATIFICATION CHECK (sample 5 species)")
  
    # Calculate deviation from ideal split
    train_pcts = []
    for species in species_counts.index:
        total = len(df[df['species'] == species])
        train_cnt = len(train_df[train_df['species'] == species])
        ideal_pct = 70  # target train percentage
        actual_pct = 100 * train_cnt / total
        deviation = abs(actual_pct - ideal_pct)
        train_pcts.append(deviation)

    print(f"Mean deviation from 70% train split: {np.mean(train_pcts):.1f}%")
    print(f"Max deviation: {np.max(train_pcts):.1f}%")
    print(f"Species with < 50% in train: {(np.array(train_pcts) > 20).sum()}")
    
    # Check for missing species in train
    print("DATA QUALITY CHECKS")
    
    train_species = set(train_df['species'])
    val_species = set(val_df['species'])
    test_species = set(test_df['species'])
    
    only_in_val = val_species - train_species
    only_in_test = test_species - train_species
    
    if only_in_val:
        print(f"  {len(only_in_val)} species appear ONLY in validation set:")
        for s in list(only_in_val)[:5]:
            print(f"     - {s}")
    else:
        print("No species isolated to validation set only")
    
    if only_in_test:
        print(f" {len(only_in_test)} species appear ONLY in test set:")
        for s in list(only_in_test)[:5]:
            print(f"     - {s}")
    else:
        print("No species isolated to test set only")
    
    # Check image availability if url_map exists
    print("IMAGE AVAILABILITY CHECK")
    
    # Try to find url_map (same logic as Main.py)
    url_map = {}
    candidates = [
        '/data/s4610601/MultiModal_images/url_to_path.csv',
        '/Volumes/PRO-G40/MultiModal_images/url_to_path.csv',
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            um = pd.read_csv(candidate)
            url_map = dict(zip(um['url'], um['local_path']))
            print(f"Found url_map: {len(url_map):,} local paths from {candidate}")
            break
    
    if url_map and 'url' in df.columns:
        df_with_local = df.copy()
        df_with_local['has_local'] = df['url'].map(lambda u: u in url_map)
        has_img = df_with_local['has_local'].sum()
        print(f"Samples with local images: {has_img:,} / {len(df):,} ({100*has_img/len(df):.1f}%)")
        
        # Check per split
        for name, split in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            split_has = split['url'].map(lambda u: u in url_map).sum()
            pct = 100 * split_has / len(split)
            print(f"  {name}: {split_has:,} / {len(split):,} ({pct:.1f}%)")
    else:
        print("No url_map found")
    
    # Save splits for future reference
    out_dir = 'split_inspection'
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(f'{out_dir}/train_split.csv', index=False)
    val_df.to_csv(f'{out_dir}/val_split.csv', index=False)
    test_df.to_csv(f'{out_dir}/test_split.csv', index=False)
    
    print("SUMMARY")
    print(f"Total samples:    {len(df):,}")
    print(f"Total species:    {df['species'].nunique():,}")
    print(f"Train samples:    {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val samples:      {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test samples:     {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"Min samples/species: {species_counts.min()}")
    print(f"Max samples/species: {species_counts.max()}")
    print(f"Median samples/species: {species_counts.median():.1f}")

if __name__ == '__main__':
    main()