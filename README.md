# Multimodal Deep Learning for Cryptic Species Classification

A deep learning pipeline that investigates whether geographic context enhances classification of cryptic species beyond image features alone. 
This repository combines visual features from ResNet-50 with five distinct geographic encodings, evaluated under early and late fusion strategies 
on the CrypticBio-Common benchmark.

## Overview

Cryptic species morphologically similar organisms that are biologically distinct present a significant challenge for automated classification, as standard vision-only models rely on visual discriminability that is by definition weak for these taxa. This project investigates whether geographic context can close that gap.

The core hypothesis follows from allopatric divergence theory: cryptic species that share morphology frequently occupy non-overlapping ranges, meaning geographic  location carries discriminative signal that image features cannot. The pipeline augments a ResNet-50 backbone with five geographic encodings under two fusion strategies, evaluated on the **CrypticBio-Common benchmark**: 158 cryptic species drawn from seven taxonomic classes (Aves, Insecta, Arachnida, Squamata, Gastropoda, Magnoliopsida, Agaricomycetes), with 15,801 georeferenced observations spanning a near-global range.

## Requirements

### System Requirements
- Python 3.8 or higher
- CUDA 11.0+ (optional, for GPU acceleration)
- pip or conda (Python package management)

### Python Libraries
- **Deep Learning & Vision**: `torch` (PyTorch), `torchvision`
- **Data Processing**: `numpy`, `pandas`, `scikit-learn`
- **Image Handling**: `Pillow` (PIL)
- **Geographic Encoding**: `h3`, `scipy`
- **Utilities**: `tqdm`, `requests`

### Optional Dependencies
- **Julia 1.6+** (for graph analysis and Makie visualisations)
- **R 3.6+** (for geographic distribution visualisations with `worldmap.R`)

## Installation

### Clone and Setup Python Environment

```bash
git clone https://github.com/luca-p27/Final_Multimodal_Group_10.git
cd Final_Multimodal_Group_10
```

### Install Dependencies

Install all required Python libraries:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision
pip install numpy pandas scikit-learn Pillow
pip install h3 scipy
pip install tqdm requests
```

### Optional: Julia Installation (for Graph Analysis)

Install Julia from https://julialang.org and add the Makie package:

```julia
using Pkg; Pkg.add("Makie")
```


## Usage

```bash
cd scripts/server
python Main.py --encoding raw --fusion early --epochs 30 --batch_size 32
```

Encoding options: `raw`, `wrap`, `sh`, `hex`, `geo_label`
Fusion options: `early`, `late`, `both`

For `geo_label` encoding, the `--geo_mode` flag controls which embeddings are used:
`--geo_mode country|continent|both` (default: `both`)



## Repository Structure

### Core Scripts

**`Dataset.py`**
Handles data loading, cleaning, and preprocessing. Returns `(image, geo, label)`
tuples for each sample. Images are loaded from local storage or downloaded via URL
with automatic caching. Supports both continuous and discrete geographic encodings.

**`Encoder.py`**
Implements all five geographic encoders. Continuous encoders (`RawEncoder`,
`WrapEncoder`, `SHEncoder`) are precomputed once at dataset initialisation.
Discrete encoders (`HexGridEncoder`, `GeoLabelEncoder`) are learnable `nn.Module`
instances trained end-to-end with the rest of the network.

**`Model.py`**
Defines the multimodal architecture, integrating visual and geographic features
under both fusion strategies.

**`Train.py`**
Training loop with validation, checkpoint management, and metrics tracking.

**`Main.py`**
Entry point for the full pipeline: model initialisation, training, evaluation,
and prediction generation.

### Utilities

**`download_images.py`**
Downloads and caches images from the iNaturalist S3 bucket.

**`Snakefile`** (`scripts/server/`)
Orchestrates the full server-side pipeline: runs all encoder x fusion combinations
and downstream analyses.


### Analysis Scripts

**`add_continent.py` & `lon-lat2country.jl`**
Utility functions for mapping GPS coordinates to country and continent labels.

**`Snakefile`** (`scripts/non-server/`)
Orchestrates the non-server analysis pipeline: runs `lon-lat2country.jl` and
`add_continent.py` to produce annotated coordinate files.

### Visualisation

**`plot_results.py`**
Generates all main and supplementary figures from model predictions and per-class
metrics produced by `Main.py`. Reads from `predictions/` and
`output/per_class_metrics/`, writes PNGs and a LaTeX table to `figures/`.

**`worldmap.R`**
Geographic distribution visualisations per taxonomic class.

**`analyses.jl` and `graph_analysis.jl`**
Julia scripts for graph analysis and Makie visualisations.


## Geographic Encodings

Five strategies map raw GPS coordinates to a fixed-length feature vector.

**Raw** — Normalises latitude by 90° and longitude by 180°, mapping both to
[−1, 1]. Output: 2-d. The simplest possible encoding, with no geometric structure.

**Wrap** — Projects each coordinate onto the unit circle as sine-cosine pairs,
following Mac Aodha et al. (2019). Output: 4-d. Preserves the circular topology
of longitude so that −180° and +180° map to the same point rather than opposite
ends of a line.

**Spherical harmonics** — Evaluates real spherical harmonic basis functions up to
degree L=10, following Rußwurm et al. (2024). Output: 121-d. Captures spatial
patterns from broad global gradients (low degree) down to fine regional variation
(high degree).

**Hex grid** — Discretises the Earth into hexagonal cells using the H3 library at
two resolutions: coarse (resolution 4) and fine (resolution 6). Each location gets
one 64-d learnable embedding per resolution; these are concatenated and projected
to 128-d. Using two resolutions lets the model capture both regional and local
spatial patterns.

**Geo-label** — Maps each observation to its country via nearest-neighbour lookup
in the GeoNames dataset, then derives the continent from the country coordinates
in geospatial space rather than by jurisdictional assignment. Country and continent
are encoded as separate 64-d learnable embeddings, concatenated and projected to
128-d.

## Fusion Strategies

The key design choice is when to combine the image and location information.

**Baseline** — Image features from ResNet-50 pass directly to the classifier,
with no geographic input.

**Early fusion** — Image and geographic features are concatenated into a single
vector before the shared classifier. The model can learn how the two signals
interact end-to-end.

**Late fusion** — Separate classification heads are trained for the image and
geographic branches. Their predictions are combined via a weighted average
(w = 1.0 each by default), so neither modality dominates out of the box.

## Plots and Additional Experiments

**Confidence calibration** — KDE plots (bandwidth 0.10) comparing confidence
distributions for correct and incorrect predictions, showing how well-calibrated
each model is.

**Graph analysis** — Predictions are interpreted as a weighted directed graph
G(V, E, W), where each edge (u, v) encodes how often the model confuses ground
truth species u with predicted species v. A better model produces more weakly
connected components, reflecting fewer cross-species confusions. Metrics and
visualisations are computed in Julia with Makie.

**Per-taxon F1 breakdown** — Per-species F1 gains relative to the image-only
baseline, aggregated by taxonomic class, to identify which taxa benefit most
from geographic fusion.

**Species-level analysis** — Per-species F1 scores joined with taxonomic and
geographic metadata, examining whether gains are largest for species whose cryptic
congeners occupy non-overlapping ranges.

## To run the scripts analyses
- Make sure that an environment using the requirements.txt exists somewhere on the server.
- And that the disk quota is not reached.
- Ensure snakemake is installed like [detailed in the documentation](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html)


### Pre-analysis (optional, local, **Please do not attempt to do this on the server**)
To reproduce the input file that is used `Snakefile_before_analysis` can be used as follows:
Ensure that [Julia](https://julialang.org) is also installed besides the requirements in requirements.txt.
Then run:

```shell
conda activate snakemake
```

```shell
snakemake --snakefile scripts/non-server/Snakefile_before_analysis --cores 1
```

And everything should run accordingly

### Running aNN models **(Use the server)**
Same requirements needed as before, except for Julia, which can be omitted.

make sure that `scripts/server/Snakefile` 
- has the correct path for `main_folder`, which is the folder in which everything from the project is stored, i.e. it should end with /final_multimodal_group_11/.
- has a correct filepath to the loaded environment that can be activated in the `environment_path` variable.
- Points to a valid GPU in the `CUDA_VISIBLE_DEVICES=0` command which is used, verify using `btop` in the shell which GPU's are available.


Then you are able to run the model using:
```shell
conda activate snakemake
```
And then running (with sufficiently many cores for downloading the images):
```shell
snakemake --snakefile "scripts/server/Snakefile" --cores 20 --resources gpu=1
```

In cases where you need to rerun everything:
```shell
snakemake --snakefile "scripts/server/Snakefile" --cores 20 --resources gpu=1 --forceall
```

### Downstream analyses (can be done locally) 
Though normally a little manual work is needed, since `None_early` and `None_late` are actually the Baseline.
The right files are already in the repo that you have downloaded, to run the analyses on these results, you can perform:

Make sure to run from the main directory in the repo (the layer of this README)

```shell
conda activate snakemake
```

```shell
snakemake --snakefile "scripts/non-server/Snakefile_after_analysis" --cores 3 --forceall
```

(--forceall in this case to force it to run, since everything is already there)






## Acknowledgements
This work uses the CrypticBio-Common benchmark and builds on:
- ResNet-50 backbone via torchvision
- Spherical harmonics via scipy
- H3 hexagonal grid system from Uber Technologies
- Graph visualisations via Makie.jl
