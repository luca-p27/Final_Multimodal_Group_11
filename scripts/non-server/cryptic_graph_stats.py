"""
cryptic_graph_stats.py

Analyzes connection patterns in three cryptic-species TSV files:
small_lookup_cryptic_group.tsv, cryptic_bidirectional.tsv, and
cryptic_transitive.tsv. Prints per-file statistics and a comparison table.

Usage:
    python cryptic_graph_stats.py
    python cryptic_graph_stats.py --input_dir /path/to/input
"""

import argparse
import csv
import os
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_INPUT_DEFAULT = os.path.abspath(os.path.join(_HERE, '..', '..', 'input'))

FILENAMES = [
    'small_lookup_cryptic_group.tsv',
    'cryptic_bidirectional.tsv',
    'cryptic_transitive.tsv',
]


def analyze_connections(filename):
    """Analyze connection patterns in a cryptic species TSV file"""

    # Load data
    edges = defaultdict(set)
    all_species = set()

    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if not row:
                continue
            species_a = row[0].strip()
            all_species.add(species_a)

            if len(row) > 1 and row[1].strip():
                cryptic_list = [x.strip() for x in row[1].split(',') if x.strip()]
                for species_b in cryptic_list:
                    edges[species_a].add(species_b)
                    all_species.add(species_b)

    # Calculate statistics
    total_species = len(all_species)
    total_directed_edges = sum(len(neighbors) for neighbors in edges.values())

    # Calculate undirected edges (unique pairs)
    undirected_pairs = set()
    for a, neighbors in edges.items():
        for b in neighbors:
            if a < b:  # Store each pair only once
                undirected_pairs.add((a, b))
    total_undirected_edges = len(undirected_pairs)

    # Calculate average degree
    avg_out_degree = total_directed_edges / total_species if total_species > 0 else 0
    avg_undirected_degree = (2 * total_undirected_edges) / total_species if total_species > 0 else 0

    # Find min, max connections
    if edges:
        max_connections = max(len(neighbors) for neighbors in edges.values())
        min_connections = min(len(neighbors) for neighbors in edges.values())
        species_with_max = [s for s, n in edges.items() if len(n) == max_connections]
    else:
        max_connections = min_connections = 0
        species_with_max = []

    # Check if bidirectional (for original file comparison)
    bidirectional_count = 0
    total_pairs = 0
    for a, neighbors in edges.items():
        for b in neighbors:
            total_pairs += 1
            if b in edges and a in edges[b]:
                bidirectional_count += 1
    bidirectional_ratio = bidirectional_count / total_pairs if total_pairs > 0 else 0

    # Print results
    print(f"FILE: {filename}")

    print(f"Total unique species: {total_species:,}")
    print(f"Total directed edges: {total_directed_edges:,}")
    print(f"Total undirected edges (unique pairs): {total_undirected_edges:,}")
    print(f"\nAverage out-degree (connections per species): {avg_out_degree:.1f}")
    print(f"Average undirected degree: {avg_undirected_degree:.1f}")
    print(f"\nMin connections for a species: {min_connections}")
    print(f"Max connections for a species: {max_connections}")
    if species_with_max:
        print(f"  Species with max: {species_with_max[0][:50]}... ({max_connections} connections)")
    print(f"\nBidirectional ratio: {bidirectional_ratio:.1%}")

    # Distribution of connection counts
    connection_counts = [len(neighbors) for neighbors in edges.values()]
    connection_counts.sort(reverse=True)
    print(f"\nTop 10 species by connection count:")
    for i, count in enumerate(connection_counts[:10], 1):
        print(f"  {i}. {count} connections")

    return {
        'species': total_species,
        'directed_edges': total_directed_edges,
        'undirected_edges': total_undirected_edges,
        'bidirectional_ratio': bidirectional_ratio
    }


def main():
    p = argparse.ArgumentParser(
        description='Analyze cryptic species connection patterns across three TSV files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input_dir', default=_INPUT_DEFAULT,
                   help='directory containing the three input TSV files')
    args = p.parse_args()

    files = [os.path.join(args.input_dir, name) for name in FILENAMES]

    # Count for each file
    results = {}
    for filepath in files:
        try:
            results[filepath] = analyze_connections(filepath)
        except FileNotFoundError:
            print(f"\n{filepath}: NOT FOUND — check the file path")

    # Compare the three files
    if len(results) == 3:
        print("COMPARISON SUMMARY")
        print(f"{'Metric':<30} {'Original':>15} {'Bidirectional':>15} {'Transitive':>15}")
        print(f"{'-'*75}")

        for metric in ['species', 'directed_edges', 'undirected_edges', 'bidirectional_ratio']:
            values = [results[f][metric] for f in files]
            if metric == 'bidirectional_ratio':
                formatted = [f"{v:.1%}" for v in values]
            elif metric == 'species':
                formatted = [f"{v:,}" for v in values]
            else:
                formatted = [f"{v:,}" for v in values]

            metric_name = metric.replace('_', ' ').title()
            print(f"{metric_name:<30} {formatted[0]:>15} {formatted[1]:>15} {formatted[2]:>15}")


if __name__ == '__main__':
    main()
