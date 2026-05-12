from __future__ import annotations

"""
make_local_tsv.py

Matches downloaded images to TSV rows by index (row 0 -> 000000.jpg, etc.) and
writes a new TSV with a local_path column. Run this after download_images.py to
give the training pipeline direct file paths instead of downloading on-the-fly.
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def open_mapping(folder):
    infile = open(f"{folder}/url_to_path.csv", "r") 
    infile.readline() # skip header
    to_local = dict()
    for line in tqdm(infile, desc="Saving Mapping"):
         items = line.split(',')
         to_local[items[0]] = items[1]
    infile.close()
    return to_local

def write_mapping(infilename, outfilename, mapping, sep='\t'):
	infile = open(infilename, "r")
	out_file = open(outfilename, "w")
	header_str = infile.readline().strip()

	try:
		url_index = header_str.split(',').index('url')
	except:
		print("url index not found, using default: 6")
		url_index = 6
	matching = 0
	missing = 0

	out_file.write(header_str + sep + "local_path\n")
	for line in tqdm(infile, desc="Matching images"):
		items = line.split(sep)
		out_str = mapping.get(items[url_index], "").strip()
		if out_str == "":
			missing += 1
		else:
			matching += 1
		out_file.write(line.strip() + sep + out_str + "\n")
	infile.close()
	out_file.close()
	return matching, missing


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='/data/Multimodal/MultiModal_Assignment/input/CrypticBio-Common_continent.tsv')
    p.add_argument('--img_dir',   default='/data/s2801973/Multimodal_images')
    p.add_argument('--out_path',  default='/data/s2801973/data/CrypticBio-Common_continent_local2.tsv')
    return p.parse_args()


def main():
	args      = parse_args()
	data_path = Path(args.data_path)
	img_dir   = Path(args.img_dir)
	out_path  = Path(args.out_path)

	sep = '\t' if data_path.suffix == '.tsv' else ','

	mapping = open_mapping(img_dir)

	matching,  missing = write_mapping(data_path, out_path, mapping, sep)


	print(f"\nMatched : {matching:,} / {(matching+missing):,}")
	print(f"Missing : {missing:,}")

	print(f"Saved   : {out_path}")
	print(f"\nNext step: python Main.py --data_path {out_path}")


if __name__ == '__main__':
    main()