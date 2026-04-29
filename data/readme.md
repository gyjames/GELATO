# GELATO Directory Overview

### `GELATO/`
Core code directory containing the main model implementation and workflow scripts.

- `main.py`: main entry script.
- `model.py`: model definition.
- `utils.py`: general utility functions.
- `Recommend_media.py`: code for growth medium recommendation.
- `Metabolite_utilization_predicte.py`: code for metabolite utilization prediction.
- `annot_genome.py`: genome annotation script.
- `annot_genome.md`: documentation for genome annotation.
- `config.yaml`: configuration file.

### `data/`
Data directory containing files used for training, evaluation, mapping, and benchmarking.

- `annot_data.csv`: master table of strain annotation information.
- `phenotypes.csv`: phenotype information table.
- `function_embedding.csv`: embedding features used for protein function prediction.
- `medium.json`, `species2medium.json`, `cluster_medium.csv`: data used for growth medium recommendation.
- `standard_name.json`: standard name mapping used in GEM analysis.
- `fastani_env_result.csv`: ANI-based medium prediction results.
- `Compound2ID.csv`: mapping between compounds and IDs.
- `Synthetic_reaction_db.json`: reaction database used for producer inference.
- `suger_acid_type.csv`: classification table for sugar and acid compounds.
- `suger acid kegg pathways.xlsx`: KEGG pathway information related to sugar and acid metabolism.

#### `data/suger_acid_KAGs/`
KAG result files for sugar- and acid-related compounds. File names follow the format `compound_KAG.json`.

#### `data/benchmark_data/`
Benchmark data directory.

Subdirectories include:

- `agora/`: benchmark files related to AGORA2.
- `auto_gem/`: benchmark files related to automatically reconstructed GEMs.
- `species_split/`: benchmark data split by species.
- `phylum_split/`: benchmark data split by phylum.

### `scripts/`

- `train_model.py`: main model training script.
- `train_rf.py`: random forest training script.
- `train_svm.py`: SVM training script.
- `Integrated_gradients.py`: script for feature attribution analysis.
- `protein_function_prediction.py`: protein function prediction script.
- `simulate_incomplete_genome.py`: script for incomplete genome simulation.
- `pred_media.ipynb`: notebook for growth medium prediction.
- `KAG_analysis.ipynb`: notebook for KAG enrichment analysis.
- `GEM_intake_simulation.ipynb`: notebook for GEM intake simulation analysis.
- `MES.ipynb`: notebook for Metabolite Exchange Score analysis.
- `glove.sh`: shell script for building word embeddings.
