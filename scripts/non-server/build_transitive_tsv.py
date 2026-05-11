"""
build_transitive_tsv.py

Reads the one-directional cryptic species lookup TSV, finds connected components
via BFS, then writes a transitive closure TSV where every species in a component
lists all other members as cryptic partners.

Usage:
    python build_transitive_tsv.py
    python build_transitive_tsv.py --input /path/to/small_lookup_cryptic_group.tsv \
                                    --output /path/to/cryptic_transitive.tsv
"""

import argparse
import csv
import os
from collections import defaultdict, deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_INPUT_DEFAULT = os.path.abspath(os.path.join(_HERE, '..', '..', 'input'))


def main():
    p = argparse.ArgumentParser(
        description='Build a transitive-closure cryptic species TSV.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input',  default=os.path.join(_INPUT_DEFAULT, 'small_lookup_cryptic_group.tsv'),
                   help='path to the one-directional input TSV')
    p.add_argument('--output', default=os.path.join(_INPUT_DEFAULT, 'cryptic_transitive.tsv'),
                   help='path for the transitive-closure output TSV')
    args = p.parse_args()

    # Step 1: build undirected graph from bidirectional data (reuse edges from above)
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

    # Step 2: find connected components
    visited = set()
    components = []

    for node in edges:
        if node not in visited:
            # BFS/DFS to get component
            queue = deque([node])
            comp = set()
            while queue:
                curr = queue.popleft()
                if curr in visited:
                    continue
                visited.add(curr)
                comp.add(curr)
                for neighbor in edges[curr]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            components.append(comp)

    # Step 3: build transitive TSV
    transitive_edges = {}
    for comp in components:
        if len(comp) == 1:
            # isolated node
            lone = comp.pop()
            transitive_edges[lone] = set()
        else:
            # all-to-all within component
            sorted_comp = sorted(comp)
            for species in sorted_comp:
                others = [x for x in sorted_comp if x != species]
                transitive_edges[species] = set(others)

    # Step 4: write output
    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for species in sorted(transitive_edges.keys()):
            cryptic_set = transitive_edges[species]
            if cryptic_set:
                writer.writerow([species, ','.join(sorted(cryptic_set))])
            else:
                writer.writerow([species, ''])

    print(f"Written: {args.output}")


if __name__ == '__main__':
    main()
