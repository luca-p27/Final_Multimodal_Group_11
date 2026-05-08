from __future__ import annotations

"""
make_local_tsv.py
-----------------
Koppelt de bestaande images in input/images/ aan de TSV
en schrijft een nieuwe TSV met local_path kolom.

Werkt omdat download_images.py de index als bestandsnaam gebruikt:
    rij 0   -> 000000.jpg / 000000.jpeg
    rij 1   -> 000001.jpg / 000001.jpeg
    etc.
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='/data/Multimodal/MultiModal_Assignment/input/CrypticBio-Common_continent.tsv')
    p.add_argument('--img_dir',   default='/data/s2801973/Multimodal_images/')
    p.add_argument('--out_path',  default='/data/s2801973/data/CrypticBio-Common_continent_local2.tsv')
    return p.parse_args()


def main():
    args      = parse_args()
    data_path = Path(args.data_path)
    img_dir   = Path(args.img_dir)
    out_path  = Path(args.out_path)

    sep = '\t' if data_path.suffix == '.tsv' else ','
    df  = pd.read_csv(data_path, sep=sep)
    print(f"Loaded {len(df):,} records from {data_path}")

    local_paths = []
    missing     = 0

    for i in tqdm(range(len(df)), desc="Matching images", unit="row"):
        # try both extensions
        found = ''
        for ext in ('.jpg', '.jpeg', '.png', '.webp'):
            candidate = img_dir / f"{i:06d}{ext}"
            if candidate.exists() and candidate.stat().st_size > 0:
                found = str(candidate)
                break
        if not found:
            missing += 1
        local_paths.append(found)

    df['local_path'] = local_paths

    n_ok = sum(1 for p in local_paths if p)
    print(f"\nMatched : {n_ok:,} / {len(df):,}")
    print(f"Missing : {missing:,}")

    df.to_csv(out_path, sep='\t', index=False)
    print(f"Saved   : {out_path}")
    print(f"\nNext step:")
    print(f"  python Hex_tuning.py --data_path {out_path}")


if __name__ == '__main__':
    main()