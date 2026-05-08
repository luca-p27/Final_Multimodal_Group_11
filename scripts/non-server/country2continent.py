import pycountry_convert as pc
import argparse


def change_file_per_line(input_filename, output_filename):
	"""
	Adds continent information per line in the input file, by using its country.
	"""
	infile = open(input_filename, 'r')
	out_file = open(output_filename, "w")
	counter = 0
	invalid_country_names = {"Reunion": "Africa", 
						     "Curacao": "South America", 
							 "Bonaire, Saint Eustatius and Saba": "South America", 
							 "Vatican": "Europe", 
							 "Palestinian Territory": "Asia", 
							 "Kosovo": "Europe",
							 "Saint Barthelemy": "South America", 
							 "U.S. Virgin Islands": "South America",
							 "Sint Maarten": "South America",
							 "Saint Helena": "Africa"}
	for line in infile:
		line = line.strip()
		if counter:
			items = [i.strip() for i in line.split('\t')]
			if items[-1] not in invalid_country_names.keys():
				country_alpha = pc.country_name_to_country_alpha2(items[-1])
				country_continent_code = pc.country_alpha2_to_continent_code(country_alpha)
				country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
				items.append(country_continent_name)
			else:
				items.append(invalid_country_names[items[-1]])
			out_file.write("\t".join(items) + "\n")
		else:
			line += "\tcontinent\n"
			out_file.write(line)
		counter += 1
	out_file.close()
	infile.close()

def parse_arguments():
    usage = """
	Adds continent data to each datarow in a .tsv datafile
	Usage:
    python %(prog)s --input-file <input_file> --output-file <output_file>
    """
    parser = argparse.ArgumentParser(description=f"{usage}", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input-file', '-i', type=str, help='fullpath name for input file', required=True)
    parser.add_argument('--output-file', '-o', type=str, help='fullpath name for output file', required=True)
    return parser.parse_args()


def main():
	args = parse_arguments()
	input_filename = args.input_file
	output_filename = args.output_file
	change_file_per_line(input_filename, output_filename)


if __name__ == "__main__":
	main()
