import csv
from collections import defaultdict

# Load data
edges = defaultdict(set)

with open('D:/02_School/MSc/02_Sem2627-2/MultiModal/MultiModal_Assignment/input/small_lookup_cryptic_group.tsv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        species_a = row[0].strip()
        if len(row) > 1 and row[1].strip():
            cryptic_list = [x.strip() for x in row[1].split(',') if x.strip()]
            for species_b in cryptic_list:
                edges[species_a].add(species_b)
                edges[species_b].add(species_a)

# Write bidirectional TSV
with open('D:/02_School/MSc/02_Sem2627-2/MultiModal/MultiModal_Assignment/input/cryptic_bidirectional.tsv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    for species, cryptic_set in sorted(edges.items()):
        if cryptic_set:
            writer.writerow([species, ','.join(sorted(cryptic_set))])
        else:
            writer.writerow([species, ''])