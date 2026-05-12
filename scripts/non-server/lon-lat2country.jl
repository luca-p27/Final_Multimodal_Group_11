try
	using ArgParse
	using ReverseGeocode
	using StaticArrays
catch
	using Pkg
	Pkg.add("ArgParse")
	Pkg.add("ReverseGeocode")
	Pkg.add("StaticArrays")
finally
	using ArgParse
	using ReverseGeocode
	using StaticArrays
end


using ReverseGeocode, StaticArrays

function change_file_per_line(input_filename, output_filename)
	#=
	Changes the raw input file to a tsv and adds a country data column to the dataset
	=#
	gc = Geocoder()
	infile = open(input_filename, "r")
	out_file = open(output_filename, "w")
	counter = 0
	in_counter = 0
	for line in eachline(infile)
		line = strip(line)
		line = replace(line, ", " => '#')
		if !iszero(counter) && length(line) > 5
			items = split(line, ',')
			if length(items[5]) > 2 && length(items[6]) > 2
				lat = parse(Float64, items[5])
				lon = parse(Float64, items[6])
				country = decode(gc, SA[lat, lon])[:country]
				push!(items, country)
				in_counter += 1
				println(out_file, join(items, '\t'))
			end
		else
			line = replace(line, "," => '\t')
			line *= "\tcountry"
			println(out_file, line)
		end
		counter += 1
	end
	println(counter)
	close(out_file)
	close(infile)
end

function parse_arguments()
	#=
	Parses arguments
	requires both an input and output file.
	=#
	s = ArgParseSettings()
	@add_arg_table s begin
		"--input-file"
			arg_type = String
			help = "Fullpath filename of the input file"
			required = true
		"--output-file"
			arg_type = String
			help = "Fullpath filename of the output file"
			required = true
	end
	return parse_args(s)
end


function main()
	parsed_args = parse_arguments()
	input_filename = parsed_args["input-file"]
	output_filename = parsed_args["output-file"]

	change_file_per_line(input_filename, output_filename)
end

main()