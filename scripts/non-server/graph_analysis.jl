try
    using Graphs
    using SimpleWeightedGraphs
    using CairoMakie
	using ArgParse
catch
	using Pkg
	Pkg.add("Graphs")
    Pkg.add("SimpleWeightedGraphs")
    Pkg.add("CairoMakie")
    Pkg.add("ArgParse")
finally
    using Graphs
    using SimpleWeightedGraphs
    using CairoMakie
	using ArgParse
end

function plot_domination_matrix(domination_matrix, model_prediction_data, plots_folder)
    #=
    Plots the domination matrix in pdf format.
    =#
    m = length(keys(domination_matrix))
    domination_matrix_static = zeros(Int, m, m)

    model_names = sort!([i for i in keys(domination_matrix)], by= x -> model_prediction_data[x]["num_correct"])
    model_names_labelled = [uppercasefirst(i) for i in model_names]
    println(model_names)

    for (i, model_name) in enumerate(model_names)
        for (j, model_name2) in enumerate(model_names)
            domination_matrix_static[j, i] = domination_matrix[model_name][model_name2]
        end
    end

    
    fig = Figure(size = (1200, 600), fontsize = 20)
    ax = Axis(fig[1, 1], xticks = (eachindex(model_names), model_names_labelled), yticks = (eachindex(model_names), model_names_labelled))
    hmap = heatmap!(ax, domination_matrix_static, colormap = :viridis)
    for i in 1:length(model_names), j in 1:length(model_names)
        txtcolor = domination_matrix_static[i, j] < 350 ? :white : :black
        text!(ax, "$(domination_matrix_static[i,j])", position = (i, j),
            color = txtcolor, align = (:center, :center))
    end
    Colorbar(fig[1, 2], hmap, width = 15, ticksize = 15)
    ax.xticklabelrotation = π / 3
    ax.xticklabelalign = (:right, :center)
    ax.xlabel = "Dominated"
    ax.ylabel = "Dominating"
    ax.title = "Domination matrix (n=$(length(model_prediction_data["Baseline"]["is_correct_list"])))"
    save("$(plots_folder)/domination_matrix.pdf", fig)
end


function get_domination_models(model_prediction_data)
    #=
    Returns the domination of each combination of comparing models with each other.
    Domination referring to being right for a particular prediction, when the other model is not.
    =#
    domination_matrix::Dict{String, Dict{String, Int}} = Dict()
    for (i, leading_model) in enumerate(keys(model_prediction_data))
        for (j, lagging_model) in enumerate(keys(model_prediction_data))
            score = 0
            for prediction_index in eachindex(model_prediction_data[leading_model]["is_correct_list"])
                prediction_value_lead = model_prediction_data[leading_model]["is_correct_list"][prediction_index]
                prediction_value_lag = model_prediction_data[lagging_model]["is_correct_list"][prediction_index]
                confidence_lead = model_prediction_data[leading_model]["confidence_list"][prediction_index]
                confidence_lag = model_prediction_data[lagging_model]["confidence_list"][prediction_index]

                if isone(prediction_value_lead) && iszero(prediction_value_lag)
                    score += 1
                elseif isone(prediction_value_lead) && isone(prediction_value_lag) && (confidence_lead > confidence_lag)
                    # score += 1
                elseif iszero(prediction_value_lead) && iszero(prediction_value_lag) && (confidence_lead < confidence_lag)
                    # score += 1
                end
            end
            if isnothing(get(domination_matrix, leading_model, nothing))
                domination_matrix[leading_model] = Dict()
            end
            domination_matrix[leading_model][lagging_model] = score
        end
    end
    println(domination_matrix)
    return domination_matrix
end


function plot_inneighborhood(model_prediction_data, plots_folder)
    #=
    Plots the in-neighborhood of mistakes for each of the models in the dataset.
    as a scatterplot with a unique marker per unique location encoding
    for early fusion, the marker is outlined with the color for the specific model and filled with white.
    =#
    f = Figure()
    ax = Axis(f[1, 1],
              title="In-neighborhood distribution of misclassifcations",
              xlabel="No. in-neighbors (mispredictions)",
              ylabel="Frequency",
              yminorticksvisible=true,
              yminorgridvisible=true)
    
    models_sorted = sort!([i for i in keys(model_prediction_data)], by=x -> model_prediction_data[x]["num_correct"])

    colors_for_model = Dict("Baseline" => "#000000",
                            "both" => "#117733",
                            "continent" => "#44AA99",
                            "country" => "#88CCEE",
                            "raw" => "#DDCC77",
                            "sh" => "#CC6677",
                            "wrap" => "#AA4499",
                            "hex" => "#882255")

    marker_for_model = Dict("Baseline" => :circle,
                            "both" => :pentagon,
                            "continent" => :dtriangle,
                            "country" => :utriangle,
                            "raw" => :rect,
                            "sh" => :xcross,
                            "wrap" => :star4,
                            "hex" => :hexagon)



    model_rename_keys = [i for i in keys(colors_for_model)]

    counter = 0
    for model in models_sorted
        model_alt = model_rename_keys[[occursin(i, model) for i in model_rename_keys]][1]

        x = [i for i in keys(model_prediction_data[model]["mistaken_inneighborhood"])]
        y = [model_prediction_data[model]["mistaken_inneighborhood"][i] for i in x]



        if occursin("early", model)
            if occursin("both", model)
                model = "Country + Cont early"
            end
            scatter!(x, y, color="#ffffff", strokecolor=colors_for_model[model_alt],
                markersize = 15, marker=marker_for_model[model_alt], strokewidth=1,
                label = uppercasefirst(model))
        else
            if occursin("both", model)
                model = "Country + Cont late"
            end
            scatter!(x, y, color=colors_for_model[model_alt], 
                    marker=marker_for_model[model_alt], markersize = 15, 
                    label = uppercasefirst(model))
        end
        counter += 1
    end
    vlines!(14, ymin=0.0, ymax=100, linestyle=:dash, color=:gray)

    f[1, 2] = Legend(f, ax, "Models", framevisible=false)
    save("$(plots_folder)/inneighborhood_distribution.pdf", f)
end


function get_bins(non_bins)
    #=
    Gets a raw list of elements and returns the frequency of each element in a dictionary
    with value => frequency
    =#
    bins::Dict{Int, Int} = Dict()
    for element in values(non_bins)
        bins[element] = get(bins, element, 0) + 1
    end
    return bins
end

function open_as_edgelist(filename)
    #=
    Opens the input filename as an edgelist
    and iterates the number of mistakes that happen with COL2 as the prediction
    =#
    infile = open(filename, "r")
    readline(infile) # skip header
    weighted_mistakes::Dict{String, Int} = Dict()
    for line in eachline(infile)
        items = split(line, ',')
        if items[1] != items[2]
            weighted_mistakes[items[2]] = get(weighted_mistakes, items[2], 0) + 1
        end
    end
    close(infile)
    return weighted_mistakes
end

function get_number_of_connections_in_cliques(n)
    #=
    Returns the number of edges/connections that can be found in a strong clique
    n(n-1) (fully strongly connected graph)
    =#
    return n * (n - 1)
end

function save_textab(model_prediction_data, output_folder)
    table_begin = join([raw"\begin{table}[h!]",
                   raw"\centering",
                   raw"\caption{Graph analysis results per model, showing the number of solved species, where each edge concerning the node is a self-connection;
                   the number of strong cliques (lower is better), number of edges in strong cliques (lower is better), maximum size of the strong cliques (lower is better), number of strongly connected components (higher is better),
                   number of weakly connected components (higher is better), density (lower is better).}",
                   raw"\begin{tabular}{p{2cm}p{1.5cm}p{1.5cm}p{1.5cm}p{1.5cm}p{1.5cm}p{1.5cm}p{1.5cm}p{1.5cm}p{1.5cm}}",
                   raw"\hline",
                   raw"\textbf{Model} & \textbf{Solved species} & \textbf{Strong cliques} & \textbf{Edges in strong cliques} & \textbf{Weights in strong cliques} & \textbf{Max size strong clique} & \textbf{Strong components} & \textbf{Weak components} & \textbf{Density} & \textbf{Correct Predictions}\\\\",
                   raw"\hline",
                   ""], '\n')
    lines = []
    sequence = sort!([i for i in keys(model_prediction_data)], by=k -> model_prediction_data[k]["num_correct"])



   for model in sequence
        keys = ["nr_solved_species", 
                "nr_strong_cliques", 
                "nr_connections_in_cliques",
                "sum_weights_in_strong_clique",
                "max_size_strong_clique",
                "nr_strongly_component", 
                "nr_weakly_component",
                "density",
                "num_correct"]
        str_line = "$(model) & " * join([model_prediction_data[model][key] for key in keys], " & ")
        str_line *= raw"\\\\"
        str_line = uppercasefirst(str_line)
        push!(lines, str_line)
   end 

   str_lines = join(lines, '\n')

   table_end = join(["",
                raw"\end{tabular}",
                raw"\label{tab:graph_analysis}",
                raw"\end{table}"], '\n')
    table_str = table_begin * str_lines * table_end

    out_file = open("$(output_folder)/graph_stats.tex", "w")
    write(out_file, table_str)
    close(out_file)
end

function get_solved_species(g)
    weak_components = weakly_connected_components(g)
    solved_species::Vector{Int} = []
    for component in weak_components
        if isone(length(component))
            append!(solved_species, component)
        end
    end
    return solved_species
end

function get_weights_in_strong_cliques(st_clique, edge_dict)
    Σ = 0
    for clique in st_clique
        for u in clique 
            for v in clique
                if u != v
                    Σ += edge_dict[u][v]
                end
            end
        end
    end
    return Σ
end

function strong_cliques(g)
    #=
    Returns the strong cliques for a particular graph
    defined as having a clique which is bidirectional for each unqiue edge in the clique.
    =#
    st_clique = []
    for clique in maximal_cliques(SimpleGraph(g))
            list_1 = copy(clique)
            list_2 = copy(clique)
            checks = []
            for i in list_1
                for j in list_2 
                    if j != i
                        push!(checks, has_edge(g, i, j))
                    end
                end
            end
            if all(checks) && length(clique) > 1
                push!(st_clique, clique)
            end
    end
    return st_clique
end

function open_as_graph(filename)
    #=
    Opens the prediction file as an edge list
    returning the simple directd graph, and id's to convert the vertex names to species names
    including whether the model is correct (as a vector) (is_correct_list)
    and its respective confidence (confidence_list)
    =#
    infile = open(filename, "r")
    readline(infile) # skip header
    edge_dict = Dict()
    species_id = Dict()
    species_id_inv = Dict()
    sources = []
    destinations = []
    id = 1
    counter = 0
    is_correct_list = []
    confidence_list = []
    for line in eachline(infile)
        items = split(line, ',')
        if iszero(get(species_id, items[1], 0))
            species_id[items[1]] = id
            species_id_inv[id] = items[1]
            id += 1
        end
        if iszero(get(species_id, items[2], 0))
            species_id[items[2]] = id
            species_id_inv[id] = items[2]
            id += 1
        end
        push!(sources, species_id[items[1]])
        push!(destinations, species_id[items[2]])

        if isnothing(get(edge_dict, species_id[items[1]], nothing))
            edge_dict[species_id[items[1]]] = Dict()
        end

        if isnothing(get(edge_dict[species_id[items[1]]], species_id[items[2]], nothing))
            edge_dict[species_id[items[1]]][species_id[items[2]]] = 0
        end
        edge_dict[species_id[items[1]]][species_id[items[2]]] += 1
        if items[1] == items[2]
            counter += 1
            push!(is_correct_list, 1)
        else
            push!(is_correct_list, 0)
        end
        push!(confidence_list, parse(Float64, items[3]))
    end
    el = Edge.([(sources[i], destinations[i]) for (i, k) in enumerate(sources)])
    close(infile)
    g = SimpleDiGraph(el)
    return g, species_id, species_id_inv, edge_dict, counter, is_correct_list, confidence_list
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
	end
	return parse_args(s)
end




function main()
    args = parse_arguments()

    main_folder = args["input-folder"]
    output_folder = args["output-folder"]
    plots_folder = args["plots-folder"]

    rename = Dict("None late" => "Baseline", "None early" => "Baseline")

    model_prediction_data = Dict()

    for (path, dirs, files) in walkdir(main_folder)
        for file in files
            if occursin("predictions", file)
                model_name = join(split(join(split(file, '_')[3:end], ' '), '.')[1:end-1])
                model_name = get(rename, model_name, model_name)
                model_prediction_data[model_name] = Dict("nr_strong_cliques" => 0,
                                                         "nr_weak_cliques" => 0,
                                                         "size_giant_strongly_component" => 0,
                                                         "size_giant_weakly_component" => 0,
                                                         "nr_strongly_component" => 0,
                                                         "nr_weakly_component" => 0,
                                                         "strong_cliques_list" => [],
                                                         "nr_solved_species" => 0,
                                                         "solved_species_list" => [],
                                                         "max_size_strong_clique" => 0,
                                                         "nr_connections_in_cliques" => 0,
                                                         "num_correct" => 0,
                                                         "density" => 0.0,
                                                         "is_correct_list" => [],
                                                         "confidence_list" => [])
                g, species_id, species_id_inv, edge_dict, num_correct, model_prediction_data[model_name]["is_correct_list"], model_prediction_data[model_name]["confidence_list"] = open_as_graph(joinpath(path, file))

                
                model_prediction_data[model_name]["num_correct"] = num_correct

                st_clique = strong_cliques(g)
                model_prediction_data[model_name]["nr_strong_cliques"] = length(st_clique)
                model_prediction_data[model_name]["max_size_strong_clique"] = maximum(length.(st_clique))
                model_prediction_data[model_name]["nr_weak_cliques"] = sum(map(x -> x > 1, length.(maximal_cliques(SimpleGraph(g)))))
                model_prediction_data[model_name]["nr_connections_in_cliques"] = sum(get_number_of_connections_in_cliques.(length.(st_clique)))
                model_prediction_data[model_name]["sum_weights_in_strong_clique"] = get_weights_in_strong_cliques(st_clique, edge_dict)

                model_prediction_data[model_name]["size_giant_strongly_component"] = maximum(length.(strongly_connected_components(g)))
                model_prediction_data[model_name]["size_giant_weakly_component"] = maximum(length.(weakly_connected_components(g)))
                model_prediction_data[model_name]["nr_strongly_component"] = length(strongly_connected_components(g))
                model_prediction_data[model_name]["nr_weakly_component"] = length(weakly_connected_components(g))
                model_prediction_data[model_name]["density"] = round(Graphs.density(g), digits=3)

                model_prediction_data[model_name]["strong_cliques_list"] = [[species_id_inv[j] for j in i] for i in st_clique]
                model_prediction_data[model_name]["nr_solved_species"] = length(get_solved_species(g))
                model_prediction_data[model_name]["solved_species_list"] = [species_id_inv[i] for i in get_solved_species(g)]

                model_prediction_data[model_name]["weighted_mistakes"] = open_as_edgelist(joinpath(path, file))
                non_bins = model_prediction_data[model_name]["weighted_mistakes"]
                model_prediction_data[model_name]["mistaken_inneighborhood"] = get_bins(non_bins)
            end
        end
    end
    save_textab(model_prediction_data, output_folder)
    plot_inneighborhood(model_prediction_data, plots_folder)
    domination_matrix = get_domination_models(model_prediction_data)
    plot_domination_matrix(domination_matrix, model_prediction_data, plots_folder)
end
main()
