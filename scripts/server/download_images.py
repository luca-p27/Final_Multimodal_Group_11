"""
download_images.py — Download all CrypticBio images to local disk on the server.

Resumable: skips files that are already downloaded.
Saves a url_to_path.csv mapping so the training pipeline can find local files
instead of downloading on-the-fly during training.

Usage (from Merged/ or project root):
  python download_images.py

  # custom paths
  python download_images.py --csv ../input/CrypticBio-Common.csv --out /data/s4610601/images
"""

import argparse
import hashlib
import os
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--csv', default=os.path.join("/data/Multimodal/MultiModal_Assignment/input/CrypticBio-Common_continent.tsv"),
                   help='path to CrypticBio TSV')
    p.add_argument('--out', default='/data/s2801973/MultiModal_images/',
                   help='directory to save images and url_to_path.csv')
    p.add_argument('--workers', type=int, default=8,
                   help='number of parallel download threads')
    p.add_argument('--timeout', type=int, default=60,
                   help='request timeout in seconds')
    p.add_argument('--retries', type=int, default=30,
                   help='number of retries per URL on failure')
    return p.parse_args()


def url_to_filename(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest() + '.jpg'


def download_one(url, out_path, timeout, retries):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout,
                             headers={'User-Agent': 'Mozilla/5.0'})
            r.raise_for_status()
            out_path.write_bytes(r.content)
            return True
        except Exception:
            if attempt < retries - 1:
                time.sleep(2)
    return False


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.csv):
        print(f"Error: CSV not found at {args.csv}")
        return

    df = pd.read_csv(args.csv, sep='\t')
    df = df.dropna(subset=['url'])
    df = df[df['url'].str.strip() != '']
    all_urls = df['url'].unique().tolist()
    print(f"Total unique URLs : {len(all_urls)}")

    already = {f.stem for f in out_dir.glob('*.jpg')}
    remaining = [u for u in all_urls if url_to_filename(u)[:-4] not in already]
    print(f"Already downloaded: {len(already)}")
    print(f"Remaining         : {len(remaining)}")

    if remaining:
        if args.workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            failed = []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(download_one, url,
                                out_dir / url_to_filename(url),
                                args.timeout, args.retries): url
                    for url in remaining
                }
                bar = tqdm(as_completed(futures), total=len(futures), desc='Downloading')
                for fut in bar:
                    url = futures[fut]
                    if not fut.result():
                        failed.append(url)
        else:
            failed = []
            for url in tqdm(remaining, desc='Downloading'):
                ok = download_one(url, out_dir / url_to_filename(url),
                                  args.timeout, args.retries)
                if not ok:
                    failed.append(url)

        print(f"\nFinished. Failed: {len(failed)}")
        if failed:
            fail_path = out_dir / 'failed_urls.txt'
            fail_path.write_text('\n'.join(failed))
            print(f"Failed URLs saved to {fail_path}")

    # always write/overwrite the mapping so it includes everything downloaded so far
    mapping = pd.DataFrame({
        'url':        all_urls,
        'local_path': [str(out_dir / url_to_filename(u)) for u in all_urls],
    })
    # only keep rows where the file actually exists
    mapping = mapping[mapping['local_path'].map(os.path.exists)]
    map_path = out_dir / 'url_to_path.csv'
    mapping.to_csv(map_path, index=False)
    print(f"Mapping saved     : {map_path}  ({len(mapping)} entries)")


if __name__ == '__main__':
    main()
