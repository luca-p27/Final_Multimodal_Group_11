try 
    using CairoMakie
    using HypothesisTests
    using ArgParse
catch
    import Pkg
    Pkg.add("CairoMakie")
    Pkg.add("HypothesisTests")
    Pkg.add("ArgParse")
finally 
    using CairoMakie
    using HypothesisTests
    using ArgParse
end

function get_label_table(m)
	m = replace(m, "late" => "(L)")
	m = replace(m, "early" => "(E)")
	m = replace(m, "sh" => "SH")
	m = replace(m, "both" => "Geo (Both)")
	m = replace(m, "continent" => "Geo (Continent)")
	m = replace(m, "country" => "Geo (Country)")
	return m
end


function get_label(m)
	m = replace(m, "late" => "(L)")
	m = replace(m, "early" => "(E)")
	m = replace(m, "sh" => "SH")
	m = replace(m, "both" => "Geo_both")
	m = replace(m, "continent" => "Geo_continent")
	m = replace(m, "country" => "Geo_country")
	return m
end

function create_movement_heatmap(model_prediction_data, plots_folder)
    #=
    Plots movement data comparison with baseline as a heatmap
    =#
    n_models = length(keys(model_prediction_data)) - 1
    movement_options = ["CC", "CI", "CO", "IC", "II", "IO", "OC", "OI", "OO"]

    movement_matrix_static = zeros(Float64, length(movement_options), n_models)

    model_names = sort!([i for i in keys(model_prediction_data)], by= x -> sum(model_prediction_data[x]["categories"]), rev=true)
    pop!(model_names)
    model_names_labelled = [uppercasefirst(get_label(i)) for i in model_names]


    for (i, model_name) in enumerate(model_names)
        for (m, movement) in enumerate(movement_options)
            options = "CIO"
            keys = [movement[1] * i for i in options]
            summie = sum([get(model_prediction_data[model_name]["movements"], i, 0.0) for i in keys])
            if iszero(summie)
                movement_matrix_static[m, i] = -Inf
            else
                movement_matrix_static[m, i] = round((get(model_prediction_data[model_name]["movements"], movement, 0.0) / summie) * 100, digits=1)
            end
        end
    end

    
    fig = Figure(size = (1200, 600), fontsize = 20)
    ax = Axis(fig[1, 1], xticks = (eachindex(movement_options), movement_options), yticks = (eachindex(model_names), model_names_labelled))
    hmap = heatmap!(ax, movement_matrix_static, colormap = :viridis)
    for i in 1:length(movement_options), j in 1:length(model_names)
        txtcolor = movement_matrix_static[i, j] < 60.0 ? :white : :black
        text!(ax, "$(movement_matrix_static[i,j])%", position = (i, j),
            color = txtcolor, align = (:center, :center))
    end
    Colorbar(fig[1, 2], hmap, width = 15, ticksize = 15)
    ax.xlabel = "Prediction of species compared to baseline"
    ax.ylabel = "Model"
    save("$(plots_folder)/movement_heatmap.pdf", fig)

end

function create_movement_table(model_prediction_data, output_folder)
    #=
    Creates the specific latex table used in the report for the different possible movements
    concerning the predictions between the baseline and the different fusion models.
    =#
    movement_options = ["CC", "CI", "CO", "IC", "II", "IO", "OC", "OI", "OO"]
    table_begin = join([raw"\begin{table}[h!]",
                   raw"\centering",
                   raw"\caption{Table containing the movement of predictions from the baseline compared to the new model, C = Correct, I = In Cryptic Group (mistake), O = Outside of Cryptic group (mistake); with the first character indicating the result in the baseline, and the second the result in the new model.}",
                   raw"\begin{tabular}{p{1.5cm} p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}}",
                   raw"\hline",
                   raw"\textbf{Model} & \textbf{CC} & \textbf{CI} & \textbf{CO} & \textbf{IC} & \textbf{II} & \textbf{IO} & \textbf{OC} & \textbf{OI} & \textbf{OO}\\\\",
                   raw"\hline",
                   ""], '\n')
    lines = []
    sequence = sort!([i for i in keys(model_prediction_data)], by=k -> sum(model_prediction_data[k]["categories"]))
    deleteat!(sequence, 1)



   for model in sequence
		model_label = uppercasefirst(get_label_table(model))
        str_line = "$(model_label) & " * join([get(model_prediction_data[model]["movements"], mov_option, 0) for mov_option in movement_options], " & ")
        str_line *= raw"\\\\"
        str_line = uppercasefirst(str_line)
        push!(lines, str_line)
   end 

   str_lines = join(lines, '\n')

   table_end = join(["",
                raw"\end{tabular}",
                raw"\label{tab:comparison_baseline}",
                raw"\end{table}"], '\n')
    table_str = table_begin * str_lines * table_end

    out_file = open("$(output_folder)/comparison_baseline.tex", "w")
    write(out_file, table_str)
    close(out_file)

end

function compare_with_baseline(model_prediction_data, cryptic_lookup)
    #=
    Compares each of the models with the baseline and returns the 'movement' of that model
    C --> Correct
    I --> In cryptic group
    O --> outside of cryptic group

    first-character --> Baseline
    second character --> Specific Fusion model
    =#
    CIO_list::Vector{Char} = []
    for model in keys(model_prediction_data)
        CIO_list = []
        for (p, prediction) in enumerate(model_prediction_data[model]["predicted"])
            if prediction == model_prediction_data[model]["ground_truth_list"][p]
                push!(CIO_list, 'C')
            elseif prediction in cryptic_lookup[model_prediction_data[model]["ground_truth_list"][p]]
                push!(CIO_list, 'I')
            else 
                push!(CIO_list, 'O')
            end
        end
        model_prediction_data[model]["CIO_list"] = CIO_list
    end

    freq_dict = Dict()

    for model in keys(model_prediction_data)
        if model != "Baseline"
            freq_dict = Dict()
            movements = model_prediction_data["Baseline"]["CIO_list"] .* model_prediction_data[model]["CIO_list"]
            for movement in movements
                freq_dict[movement] = get(freq_dict, movement, 0) + 1
            end
            model_prediction_data[model]["movements"] = freq_dict
        end
    end
    return model_prediction_data
end

function plot_confidence_predictions(model_prediction_data)
    #=
    (not used in the paper)
    Plots a boxplot with the confidence for mistaken and correct predictions for each model.
    =#
    x_axis_ticks = repeat(["0", "1"], length(keys(model_prediction_data)))
    f = Figure()
    ax = Axis(f[1, 1],
		 title = "Boxplot Prediction Confidence versus Ground Truth per model",
		 xlabel = "Prediction equal to Ground Truth?",
		 ylabel = "Confidence",
         xticks = (0:(1 + 2 * (length(keys(model_prediction_data)) - 1)), x_axis_ticks),
		 yticks = -0.1:0.1:1.1,
		 yminorticks = IntervalsBetween(4),
		 yminorticksvisible = true,
		 yminorgridvisible = true
	)
    sequence = sort!([i for i in keys(model_prediction_data)], by=k -> sum(model_prediction_data[k]["categories"]))
    for (index, model) in enumerate(sequence)
        boxplot!(model_prediction_data[model]["categories"] .+ 2 * Float64(index - 1), model_prediction_data[model]["values"], label = model)
    end
    f[1, 2] = Legend(f[1, 2], ax, "Model used", framevisible = false)
    save("plots/confidence_predictions.png", f)
end


function get_cryptic_rate(raw, cryptic_lookup)
    #=
    Cryptic categories shows whether misprediction is part of cryptic group (1) or not (0)
    The corresponding values are the confidence the model has in its prediction
    =#
    cryptic_categories::Vector{Int} = []
    cryptic_values::Vector{Float64} = []
    for items in raw
        if items[1] != items[2] # misprediction
            if items[2] ∉ cryptic_lookup[items[1]] # misprediction is not in cryptic group
                push!(cryptic_categories, 0)
            else
                push!(cryptic_categories, 1)
            end
            push!(cryptic_values, parse(Float64, items[3]))
        end
    end
    return cryptic_categories, cryptic_values
end

function get_prediction_rate(raw)
    #=
    categories is a vector containing whether the model is correct
    values returns the confidence of the prediction of the model
    =#
    categories::Vector{Int} = []
    values::Vector{Float64} = []
    for items in raw
        push!(categories, items[1] == items[2])
        push!(values, parse(Float64, items[3]))
    end
    return categories, values
end

function open_file(filename)
    #=
    Opens a csv file raw and returns it as a 2D vector containing a 
    vector of elements for each row.
    =#
    infile = open(filename, "r")
    readline(infile) # skip header
    raw = [split(i, ',') for i in readlines(infile)]
    close(infile)
    return raw
end

function get_cryptic_groups(filename)
    #=
    Returns the cryptic group for each respective species in a reference file
    =#
    infile = open(filename, "r")
    cryptic_lookup::Dict{String, Set{String}} = Dict()
    for line in eachline(infile)
        items = split(line, '\t')
        cryptic_lookup[items[1]] = Set(split(items[2], ','))
    end
    return cryptic_lookup
end

function parse_arguments()
	#=
	Parses arguments
	requires an input, output and plots folder
	=#
	s = ArgParseSettings()
	@add_arg_table s begin
		"--input-folder"
			arg_type = String
			help = "Fullpath filename of the input folder"
			required = true
		"--output-folder"
			arg_type = String
			help = "Fullpath filename of the output folder"
			required = true
		"--plots-folder"
			arg_type = String
			help = "Fullpath filename of the plots folder"
			required = true
        "--cryptic-path"
			arg_type = String
			help = "Fullpath name to the cryptic look-up table"
			required = true
	end
	return parse_args(s)
end



function main()
    args = parse_arguments()
    
    main_folder = args["input-folder"]
    output_folder = args["output-folder"]
    plots_folder = args["plots-folder"]

    cryptic_filename = args["cryptic-path"]
    
    cryptic_lookup = get_cryptic_groups(cryptic_filename)
    rename = Dict("None late" => "Baseline", "None early" => "Baseline")

    model_prediction_data = Dict()

    for (path, dirs, files) in walkdir(main_folder)
        for file in files
            if occursin("predictions", file)
                model_name = join(split(join(split(file, '_')[3:end], ' '), '.')[1:end-1])
                model_name = get(rename, model_name, model_name)
                raw = open_file(joinpath(path, file))

                ground_truth = [i[1] for i in raw]
                predicted = [i[2] for i in raw]




                categories, values = get_prediction_rate(raw)
                cryptic_categories, cryptic_values = get_cryptic_rate(raw, cryptic_lookup)
                # 

                tempi_1 = [iszero(i) for i in cryptic_categories]
                avg_not_cryptic = sum(tempi_1 .* cryptic_values) / length(tempi_1)

                avg_cryptic = sum(cryptic_categories .* cryptic_values) / sum(cryptic_categories)
                model_prediction_data[model_name] = Dict()
                model_prediction_data[model_name]["categories"] = categories
                model_prediction_data[model_name]["values"] = values
                model_prediction_data[model_name]["cryptic_categories"] = cryptic_categories
                model_prediction_data[model_name]["cryptic_values"] = cryptic_values
                model_prediction_data[model_name]["ground_truth_list"] = ground_truth
                model_prediction_data[model_name]["predicted"] = predicted
            end
        end
    end

    

    compare_with_baseline(model_prediction_data, cryptic_lookup)
    create_movement_table(model_prediction_data, output_folder)
    create_movement_heatmap(model_prediction_data, plots_folder)
end

main()