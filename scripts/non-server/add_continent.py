"""
add_continent.py

Adds a continent column to a TSV by looking up each row's country name.
Handles a small set of edge-case country names that pycountry_convert
does not recognise out of the box.

Usage:
    python add_continent.py --input-file input.tsv --output-file output.tsv
"""

import argparse
import pycountry_convert as pc


INVALID_COUNTRY_NAMES = {
    "Reunion": "Africa",
    "Curacao": "South America",
    "Bonaire, Saint Eustatius and Saba":"South America",
    "Vatican": "Europe",
    "Palestinian Territory": "Asia",
    "Kosovo": "Europe",
    "Saint Barthelemy": "South America",
    "U.S. Virgin Islands": "South America",
    "Sint Maarten": "South America",
    "Saint Helena": "Africa",
}


def change_file_per_line(input_filename, output_filename):
    """Adds a continent column to each data row using its country field."""
    counter = 0
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as out_file:
        for line in infile:
            line = line.strip()
            if counter:
                items = [i.strip() for i in line.split('\t')]
                if items[-1] not in INVALID_COUNTRY_NAMES:
                    country_alpha = pc.country_name_to_country_alpha2(items[-1])
                    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha)
                    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
                    items.append(country_continent_name)
                else:
                    items.append(INVALID_COUNTRY_NAMES[items[-1]])
                out_file.write("\t".join(items) + "\n")
            else:
                line += "\tcontinent\n"
                out_file.write(line)
            counter += 1


def parse_arguments():
    usage = """
    Adds continent data to each data row in a .tsv datafile
    Usage:
    python %(prog)s --input-file <input_file> --output-file <output_file>
    """
    parser = argparse.ArgumentParser(
        description=f"{usage}",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--input-file',  '-i', type=str, required=True,
                        help='full path to input file')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='full path to output file')
    return parser.parse_args()


def main():
    args = parse_arguments()
    change_file_per_line(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
