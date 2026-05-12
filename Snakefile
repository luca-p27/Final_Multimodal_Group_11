main_folder = "/home/nessinmorse/Programming/Python/Multimodal_final/Final_Multimodal_Group_11"

student_number = "s2801973"
#environment_name = ["multimodal_shite/"]


DATASETS = ["CrypticBio-Common"]
# DATASETS = ["CrypticBio-Common", "CrypticBio-CommonUnseen", "CrypticBio-Endangered", "CrypticBio-Invasive"]


SEED = 42


# --encoding all
ENCODINGS = ["wrap", "raw", "sh", "hex", "None"]
ENCODINGS_2 = ["geo_label"]
GEO_MODES = ["country", "continent", "both"]
# ENCODINGS = ["hex"]

# GEO_ARGUMENTS = ["country", "continent", "both"]
FUSION_TYPE = ["early", "late"]
# FUSION_TYPE = ["late"]

# Need separate for baseline & baseline+naive


rule all:
	input:
		expand("{main}/input/{dataset}_continent_local2.tsv",
			dataset=DATASETS,
			main=main_folder,
			),
		expand("{main_folder}/raw_models/{dataset}/{encoding}/best_model_{encoding}_{fusion}.pth",
			dataset=DATASETS,
			encoding=ENCODINGS,
			fusion=FUSION_TYPE,
			main_folder=main_folder,
			),
		expand("{main_folder}/raw_models/{dataset}/{encoding}/{geo_mode}/best_model_{encoding}_{fusion}.pth",
			dataset=DATASETS,
			encoding=ENCODINGS_2,
			fusion=FUSION_TYPE,
			geo_mode=GEO_MODES,
			main_folder=main_folder,
			)

rule download_images:
	output:
		"{main_folder}/input/{dataset}_continent_local2.tsv"
	wildcard_constraints:
		main=main_folder,
		dataset="|".join(DATASETS)
	threads: 6 
	resources:
		gpu=1
	conda:
		"envs/global_env.yml"
	shell:
		"""
		# python3 {main_folder}/scripts/server/download_images.py --csv "{main_folder}/input/{wildcards.dataset}_continent.tsv" --out "{main_folder}/images/{wildcards.dataset}" --workers {threads} --timeout 1 --retries 1
		python3 {main_folder}/scripts/server/make_local_tsv_2.py --data_path "{main_folder}/input/CrypticBio-Common_continent.tsv" --img_dir "{main_folder}/images/{wildcards.dataset}" --out_path "{main_folder}/input/{wildcards.dataset}_continent_local2.tsv"
		"""


rule run_geomodels:
	input:
		"{main}/input/{dataset}_continent_local2.tsv"
	output:
		"{main}/raw_models/{dataset}/{encoding}/{geo_mode}/best_model_{encoding}_{fusion}.pth",
		"{main}/raw_models/{dataset}/{encoding}/{geo_mode}/confusion_matrix_{encoding}_{fusion}.csv",
		"{main}/raw_models/{dataset}/{encoding}/{geo_mode}/metrics_{encoding}_{fusion}.json",
		"{main}/raw_models/{dataset}/{encoding}/{geo_mode}/per_class_metrics_{encoding}_{fusion}.csv",
		"{main}/raw_models/{dataset}/{encoding}/{geo_mode}/test_predictions_{encoding}_{fusion}.csv"
	resources:
		gpu=1
	wildcard_constraints:
		dataset="|".join(DATASETS),
		encoding="|".join(ENCODINGS_2),
		fusion="|".join(FUSION_TYPE),
		geo_mode="|".join(GEO_MODES),
		main=main_folder,
		student=student_number
	conda:
		"envs/global_env.yml"
	shell:
		"""
		CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 {wildcards.main}/scripts/server/Main.py --data_path {input} --top_n 0 --seed 42 --geo_mode {wildcards.geo_mode} --encoding {wildcards.encoding} --fusion {wildcards.fusion} --epochs 1 --out_dir /{wildcards.main}/raw_models/{wildcards.dataset}/{wildcards.encoding}/{wildcards.geo_mode}/ --seed {SEED} --url_map {wildcards.main}/images/{wildcards.dataset}/url_to_path.csv
		"""


rule run_analyses:
	input:
		"{main}/input/{dataset}_continent_local2.tsv"
	output:
		"{main}/raw_models/{dataset}/{encoding}/best_model_{encoding}_{fusion}.pth",
		"{main}/raw_models/{dataset}/{encoding}/confusion_matrix_{encoding}_{fusion}.csv",
		"{main}/raw_models/{dataset}/{encoding}/metrics_{encoding}_{fusion}.json",
		"{main}/raw_models/{dataset}/{encoding}/per_class_metrics_{encoding}_{fusion}.csv",
		"{main}/raw_models/{dataset}/{encoding}/test_predictions_{encoding}_{fusion}.csv"
	resources:
		gpu=1
	wildcard_constraints:
		dataset="|".join(DATASETS),
		encoding="|".join(ENCODINGS),
		fusion="|".join(FUSION_TYPE),
		main=main_folder
	conda:
		"envs/global_env.yml"
	shell:
		"""
		CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 {wildcards.main}/scripts/server/Main.py --data_path {input} --top_n 0 --seed 42 --encoding {wildcards.encoding} --fusion {wildcards.fusion} --epochs 1 --out_dir /{wildcards.main}/raw_models/{wildcards.dataset}/{wildcards.encoding}/ --seed {SEED} --url_map {wildcards.main}/images/{wildcards.dataset}/url_to_path.csv
		"""