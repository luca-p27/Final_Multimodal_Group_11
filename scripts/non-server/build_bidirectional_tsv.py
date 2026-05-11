"""
build_bidirectional_tsv.py

Reads the one-directional cryptic species lookup TSV and writes a bidirectional
version where every (A, B) edge also has a (B, A) edge.

Usage:
    python build_bidirectional_tsv.py
    python build_bidirectional_tsv.py --input /path/to/small_lookup_cryptic_group.tsv \
                                       --output /path/to/cryptic_bidirectional.tsv
"""

import argparse
import csv
import os
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_INPUT_DEFAULT = os.path.abspath(os.path.join(_HERE, '..', '..', 'input'))


def main():
    p = argparse.ArgumentParser(
        description='Build a bidirectional cryptic species TSV.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input',  default=os.path.join(_INPUT_DEFAULT, 'small_lookup_cryptic_group.tsv'),
                   help='path to the one-directional input TSV')
    p.add_argument('--output', default=os.path.join(_INPUT_DEFAULT, 'cryptic_bidirectional.tsv'),
                   help='path for the bidirectional output TSV')
    args = p.parse_args()

    # Load data
    edges = defaultdict(set)

    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            species_a = row[0].strip()
            if len(row) > 1 and row[1].strip():
                cryptic_list = [x.strip() for x in row[1].split(',') if x.strip()]
                for species_b in cryptic_list:
                    edges[species_a].add(species_b)
                    edges[species_b].add(species_a)

    # Write bidirectional TSV
    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for species, cryptic_set in sorted(edges.items()):
            if cryptic_set:
                writer.writerow([species, ','.join(sorted(cryptic_set))])
            else:
                writer.writerow([species, ''])

    print(f"Written: {args.output}")


if __name__ == '__main__':
    main()
