# TabHGIF: A Unified Hypergraph Influence Framework for Efficient Unlearning in Tabular Data

Official code repository for the paper:

**TabHGIF: A Unified Hypergraph Influence Framework for Efficient Unlearning in Tabular Data**

This repository implements **TabHGIF**, a hypergraph influence-function-based framework for efficient **post-hoc machine unlearning** on tabular data. The codebase supports multiple unlearning granularities, multiple hypergraph neural network backbones, tabular blind unlearning baselines, fine-tuning baselines, retraining/partial-retraining baselines, and privacy evaluation via MIA.

---

## 1. Project Overview

TabHGIF targets **machine unlearning for hypergraph-structured tabular models**.
When a deletion request is issued, the model may need to adapt to:

- **hypergraph structural changes** (edited hypergraph after deletion), and
- **parameter changes** (to remove deleted-data influence while preserving utility)

Instead of full retraining, TabHGIF performs localized hypergraph editing followed by **HGIF-based parameter correction** to efficiently approximate the retraining effect.

### Supported Unlearning Settings

- **Row unlearning** (instance-level deletion)
- **Column unlearning** (feature-level deletion)
- **Value-based / row-group unlearning** (deletion of rows selected by specific values or value combinations)

### Hypergraph Backbones

- **HGNN**
- **HGNNP**
- **HGCN**
- **HGAT**

### Baselines / Evaluation Included

- **Full Retraining**
- **Partial Retraining** (in selected scripts/settings)
- **Fine-Tuning baselines** (ACI / Bank / Credit supplementary comparisons)
- **RELOAD (TabNet backbone)** as tabular blind unlearning baseline
- **Membership Inference Attack (MIA)** evaluation
- Utility / efficiency / privacy metrics (ACC, F1, runtime, MIA AUC, forget-related accuracy, etc.)

---

## 2. Repository Structure (Aligned with Current Codebase)

```text
TabHGIF/
├── ACI_run/                                  # ACI (Adult Census Income) experiments
│   ├── HGAT/
│   │   ├── HGAT_Unlearning_col.py
│   │   ├── HGAT_Unlearning_row_nei.py
│   │   └── HGAT_Unlearning_row_Value.py
│   ├── HGCN/
│   │   ├── HGCN_PartialRetrain_row_nei.py
│   │   ├── HGCN_retrain.py
│   │   ├── HGCN_Retrain_Column.py
│   │   ├── HGCN_Unlearning_col.py
│   │   ├── HGCN_Unlearning_row_nei.py
│   │   └── HGCN_Unlearning_row_Value.py
│   ├── HGNN/
│   │   ├── HGNN_PartialRetrain_row.py
│   │   ├── HGNN_Unlearning_col.py
│   │   ├── HGNN_Unlearning_row_nei.py
│   │   ├── HGNN_Unlearning_Value.py
│   │   └── HGNNandHGNNP_Retrain_Col.py
│   └── HGNNP/
│       ├── HGNNP_PartialRetrain_row.py
│       ├── HGNNP_Unlearning_col.py
│       ├── HGNNP_Unlearning_row_nei.py
│       └── HGNNP_Unlearning_row_Value.py
│
├── bank/                                     # Bank dataset experiments
│   ├── FT_bank/                              # Fine-tuning baselines on Bank
│   │   ├── FT_HGAT/
│   │   │   ├── col.py
│   │   │   └── row.py
│   │   ├── FT_HGCN/
│   │   ├── FT_HGNN/
│   │   └── FT_HGNNP/
│   ├── HGAT/
│   │   ├── __init__.py
│   │   ├── balancedtry.py
│   │   ├── config.py
│   │   ├── data_preprocessing_Balancetry.py
│   │   ├── data_preprocessing_bank.py
│   │   ├── data_preprocessing_bank_col.py
│   │   ├── data_preprocessing_bank_retrain.py
│   │   ├── GIF_HGAT_COL.py
│   │   ├── GIF_HGAT_ROW_NEI.py
│   │   ├── HGAT_new.py
│   │   ├── HGAT_Unlearning_col.py
│   │   ├── HGAT_Unlearning_col_retrain.py
│   │   ├── HGAT_Unlearning_row_nei.py
│   │   ├── HGAT_Unlearning_row_Value.py
│   │   └── MIA_HGAT.py
│   ├── HGCN/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── data_preprocessing_bank.py
│   │   ├── data_preprocessing_col_bank.py
│   │   ├── data_preprocessing_col_retrain.py
│   │   ├── GIF_HGCN_COL_bank.py
│   │   ├── GIF_HGCN_ROW_bank.py
│   │   ├── HGCN.py
│   │   ├── HGCN_deletion.py
│   │   ├── HGCN_Unlearning_col_bank.py
│   │   ├── HGCN_Unlearning_col_retrain.py
│   │   ├── HGCN_Unlearning_row_bank.py
│   │   ├── HGCN_Unlearning_row_bank_Value.py
│   │   ├── HGCN_utils.py
│   │   └── MIA_HGCN.py
│   ├── HGNN/
│   ├── HGNNP/
│   ├── __init__.py
│   ├── GIF_HGCN_ROW.py
│   ├── HGCN_bank.py
│   ├── HGCN_Train.py
│   └── HGCN_utils.py
│
├── Baseline_FT_ACI/                          # Fine-tuning baselines on ACI
│   ├── HGAT_baseline_FT/
│   │   ├── run_ft_hgat_col_zero.py
│   │   └── run_ft_hgat_row_zero.py
│   ├── HGCN_baseline_FT/
│   ├── HGNN_baseline_FT/
│   └── HGNNP_baseline_FT/
│
├── baseline_Tabnet/                          # RELOAD (TabNet) baselines and TabNet-related scripts
│   ├── Baseline_RELOAD_ACI.py
│   ├── Baseline_RELOAD_ACI_COL.py
│   ├── Baseline_RELOAD_ACI_deletion.py
│   ├── Baseline_RELOAD_ACI_Value.py
│   ├── Baseline_RELOAD_Bank.py
│   ├── Baseline_RELOAD_Bank_col.py
│   ├── Baseline_RELOAD_Bank_deletion.py
│   ├── Baseline_RELOAD_Bank_Value.py
│   ├── Baseline_RELOAD_Credit.py
│   ├── Baseline_RELOAD_Credit_col.py
│   ├── Baseline_RELOAD_Credit_value.py
│   ├── MIA_Tabnet.py
│   ├── reload_unlearning.py
│   ├── reload_unlearning_beifen.py
│   ├── tabnetbaseline.py
│   ├── tabnetbaseline_aci.py
│   └── tabnetbaseline_bank.py
│
├── Credit/                                   # Credit dataset experiments
│   ├── credit_data/
│   ├── FT_baseline_Credit/                   # Fine-tuning baselines on Credit
│   │   ├── HGAT/
│   │   ├── HGCN/
│   │   ├── HGNN/
│   │   └── HGNNP/
│   ├── HGAT/
│   ├── HGCN/
│   ├── HGNN/
│   ├── HGNNP/
│   ├── data_preprocessing_credit.py
│   ├── HGCN.py
│   └── HGCN_run.py
│
├── database/                                 # Processed data and preprocessing artifacts
│   ├── data/
│   ├── data_banking/
│   ├── data_covertype/
│   └── data_preprocessing/
│
├── GIF/                                      # Core HGIF / TabHGIF modules (shared implementation)
│   ├── __init__.py
│   ├── GIF_HGAT_COL.py
│   ├── GIF_HGAT_ROW_NEI.py
│   ├── GIF_HGCN_COL.py
│   ├── GIF_HGCN_ROW.py
│   ├── GIF_HGCN_ROW_NEI.py
│   ├── GIF_HGNN_COL.py
│   ├── GIF_HGNN_ROW.py
│   ├── GIF_HGNN_ROW_NEI.py
│   ├── GIF_HGNNP_COL.py
│   ├── GIF_HGNNP_ROW.py
│   └── GIF_HGNNP_ROW_NEI.py
│
├── HGNNs_Model/                              # Hypergraph model implementations
│   ├── HGAT/
│   ├── HGCN/
│   │   ├── __init__.py
│   │   └── HyperGCN.py
│   ├── HGNN/
│   ├── HGNNP/
│   └── __init__.py
│
├── MIA/                                      # Shared MIA evaluation modules
│   ├── __init__.py
│   ├── MIA_HGAT.py
│   ├── MIA_HGCN.py
│   ├── MIA_HGNNP.py
│   └── MIA_utils.py
│
├── utils/
│   └── common_utils.py
│
├── config.py
├── config_HGCN.py
└── Readme
3. Naming Convention and Task Mapping

The script name usually indicates its role directly.

Unlearning Scripts

*_Unlearning_col.py
Column unlearning (feature-level deletion)

*_Unlearning_row_nei.py
Row unlearning (instance-level deletion, neighbor-based row setting)

*_Unlearning_row_Value.py or *_Unlearning_Value.py
Value-based / row-group unlearning (delete rows selected by feature values)

Baseline Scripts

*Retrain*.py
Full retraining baseline (or retraining-related comparison)

*PartialRetrain*.py
Partial retraining baseline

FT_* or Baseline_FT_*
Fine-tuning baselines used for supplementary/revision experiments

Privacy Evaluation

MIA_*.py
Membership Inference Attack (MIA) evaluation

4. Code-to-Paper Mapping (How This Matches the Paper)

This repository is organized to align with the experiments in the TabHGIF paper and its supplementary analyses.

4.1 Main TabHGIF Experiments (Hypergraph Backbones)

Folders: ACI_run/, bank/, Credit/

Backbones: HGNN / HGNNP / HGCN / HGAT

Tasks: row / column / value-based unlearning

Purpose: main utility, runtime, and privacy evaluations of TabHGIF

4.2 Retraining and Partial-Retraining Comparisons

Examples:

ACI_run/HGCN/HGCN_retrain.py

ACI_run/HGCN/HGCN_Retrain_Column.py

ACI_run/HGCN/HGCN_PartialRetrain_row_nei.py

ACI_run/HGNN/HGNN_PartialRetrain_row.py

ACI_run/HGNNP/HGNNP_PartialRetrain_row.py

Purpose: oracle and practical comparison baselines for contextualizing efficiency and utility

4.3 Fine-Tuning Baselines (Supplementary / Appendix Experiments)

Folders:

Baseline_FT_ACI/

bank/FT_bank/

Credit/FT_baseline_Credit/

Purpose: practical post-edit adaptation baselines (e.g., head-only or budgeted fine-tuning settings)

4.4 RELOAD (TabNet) Baselines for Tabular Blind Unlearning

Folder: baseline_Tabnet/

Purpose: tabular blind unlearning comparison using RELOAD-style method with TabNet backbone

Coverage: ACI / Bank / Credit, including row / column / value-based settings

4.5 MIA Evaluation

Folders: MIA/, dataset-specific MIA_*.py, and baseline_Tabnet/MIA_Tabnet.py

Purpose: privacy evaluation using membership inference attacks on hypergraph and TabNet baselines

5. Core Modules
GIF/ (Shared HGIF / TabHGIF Core)

This folder contains the shared core implementation of HGIF-based post-hoc parameter correction used by TabHGIF.

It includes backbone- and task-specific modules such as:

GIF_HGCN_ROW_NEI.py

GIF_HGCN_COL.py

GIF_HGAT_ROW_NEI.py

GIF_HGAT_COL.py

GIF_HGNN_ROW_NEI.py

GIF_HGNNP_ROW_NEI.py
etc.

These modules implement the core influence-based update logic after hypergraph editing.

HGNNs_Model/

Contains model definitions for the hypergraph backbones used in the paper:

HGAT

HGCN (including HyperGCN.py)

HGNN

HGNNP

MIA/

Contains reusable Membership Inference Attack code and utilities:

backbone-specific MIA scripts

MIA_utils.py

database/

Contains processed datasets and preprocessing-related data folders.
This is the main local data resource used by the experiments.

utils/common_utils.py

Shared utility functions used across scripts (helpers, common processing, evaluation support, etc.).

6. Datasets

The repository supports experiments on multiple tabular datasets, including:

ACI / Adult Census Income

Bank Marketing

Credit Approval

(additional processed data folders such as data_covertype are also present in database/)

Data Organization

database/data/

database/data_banking/

database/data_covertype/

database/data_preprocessing/

Credit/credit_data/

Notes

The database/ folder contains already processed data used in many experiments.

Some dataset-specific folders (e.g., Credit/) also include their own preprocessing scripts.

Some scripts may still assume local paths from the original development environment. Check configuration before running.

7. Environment Requirements

This project was developed in Python + PyTorch environments, and the experiments were commonly run in Conda environments such as `tabhgif` and `hytrel`. We recommend using **Python 3.8** with **PyTorch** (GPU version recommended), together with standard scientific Python packages, including `numpy`, `pandas`, `scipy`, `scikit-learn`, `tqdm`, `openpyxl`, and `matplotlib`. For the RELOAD (TabNet) baselines in `baseline_Tabnet/`, please also install `pytorch-tabnet`.

### Example Setup (Conda)

```bash
conda create -n tabhgif python=3.8 -y
conda activate tabhgif

# Install PyTorch according to your CUDA version
# (replace with the correct command for your machine)
# Example:
# pip install torch==1.11.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

pip install numpy pandas scipy scikit-learn tqdm openpyxl matplotlib
pip install pytorch-tabnet

## 8. Configuration

The codebase uses a mix of global config files, backbone/dataset-specific config files, and script-local settings. Before running experiments, please check the corresponding configuration files
(e.g., `config.py`, `config_HGCN.py`, and dataset-specific configs such as `bank/*/config.py`) and verify key settings including dataset paths, random seeds, deletion ratio/setting, training hyperparameters
(e.g., epochs and learning rate), HGIF iteration/damping parameters (if applicable), and GPU device selection.

9. Quick Start (Examples)

The exact behavior depends on each script and local configuration.
Please verify paths/configs before running.

9.1 ACI: HGCN Row Unlearning
python ACI_run/HGCN/HGCN_Unlearning_row_nei.py
9.2 ACI: HGCN Column Unlearning
python ACI_run/HGCN/HGCN_Unlearning_col.py
9.3 ACI: HGCN Value-Based / Row-Group Unlearning
python ACI_run/HGCN/HGCN_Unlearning_row_Value.py
9.4 ACI: HGCN Row Partial Retraining
python ACI_run/HGCN/HGCN_PartialRetrain_row_nei.py
9.5 ACI: HGCN Full Retraining (Row / Main Retrain)
python ACI_run/HGCN/HGCN_retrain.py
9.6 ACI: HGCN Column Retraining
python ACI_run/HGCN/HGCN_Retrain_Column.py
9.7 ACI: HGAT Row Unlearning
python ACI_run/HGAT/HGAT_Unlearning_row_nei.py
9.8 ACI: HGAT Column Unlearning
python ACI_run/HGAT/HGAT_Unlearning_col.py
9.9 RELOAD Baseline on ACI (TabNet)
python baseline_Tabnet/Baseline_RELOAD_ACI.py
9.10 RELOAD Column Unlearning on ACI
python baseline_Tabnet/Baseline_RELOAD_ACI_COL.py
9.11 RELOAD Value-Based Unlearning on ACI
python baseline_Tabnet/Baseline_RELOAD_ACI_Value.py
9.12 RELOAD Baseline on Bank / Credit
python baseline_Tabnet/Baseline_RELOAD_Bank.py
python baseline_Tabnet/Baseline_RELOAD_Credit.py
9.13 ACI Fine-Tuning Baseline (Example: HGAT)
python Baseline_FT_ACI/HGAT_baseline_FT/run_ft_hgat_row_zero.py
python Baseline_FT_ACI/HGAT_baseline_FT/run_ft_hgat_col_zero.py
9.14 Bank Fine-Tuning Baseline (Example: HGAT)
python bank/FT_bank/FT_HGAT/row.py
python bank/FT_bank/FT_HGAT/col.py
10. Typical Outputs and Metrics

Depending on the script, the logs typically report utility, privacy, and efficiency metrics, including test accuracy/F1 (e.g., test_acc, ACC, f1, retain_acc, forget_acc),
 membership inference attack results (e.g., mia_overall, mia_deleted, MIA AUC, and sometimes attack F1), and runtime statistics (e.g., edit_time, update_time, total_time, and
 retraining vs. unlearning time; some supplementary analyses also include runtime decomposition). For repeated experiments, results are commonly summarized as mean ± std,
 following the reporting style used in the paper.

11. Recommended Usage Order (for New Users)

If you are new to this codebase, we recommend starting with a single ACI hypergraph unlearning script (e.g., ACI_run/HGCN/HGCN_Unlearning_row_nei.py),
 then running the corresponding retraining baseline (e.g., ACI_run/HGCN/HGCN_retrain.py) for comparison, followed by a RELOAD (TabNet) baseline (e.g., baseline_Tabnet/Baseline_RELOAD_ACI.py).
 After that, you can run MIA evaluation using the shared MIA/ modules or dataset-specific MIA_*.py scripts, and then explore the extended experiments,
 including fine-tuning and partial-retraining baselines as well as Bank/Credit settings (e.g., Baseline_FT_ACI/, bank/FT_bank/, and Credit/FT_baseline_Credit/).

12. Reproducibility Notes

Please note that some scripts may still contain local path assumptions from the original development environment, so dataset and output paths should be verified before running.
In addition, key experimental settings (e.g., deletion ratio/type, epochs, learning rate, random seeds, backbone-specific parameters, and HGIF iteration/damping settings) should be checked in the corresponding configs/scripts.
GPU memory usage may vary with the dataset, backbone, deletion setting, and MIA evaluation mode, so batch size or runtime-related settings may need to be adjusted accordingly.

13. Notes on Terminology Used in This Repo
Row / Column / Value Unlearning

Row = instance/sample deletion

Column = feature deletion

Value = deleting rows selected by attribute values (value-based or row-group deletion)

RELOAD on TabNet

In this repository, RELOAD baselines are implemented in baseline_Tabnet/ using a TabNet backbone for tabular blind unlearning comparison.

FT Baselines

FT-related folders contain fine-tuning baselines used to compare practical post-edit adaptation strategies against TabHGIF in supplementary/revision experiments.

14. Citation

If you use this code in your research, please cite the corresponding paper.

@article{TabHGIF2026,
  title   = {TabHGIF: A Unified Hypergraph Influence Framework for Efficient Unlearning in Tabular Data},
  author  = {<Author List>},
  journal = {<Venue / Under Review>},
  year    = {2026}
}
15. Contact

For questions about the code, experiments, or reproduction details, please contact:

Name: <Your Name>

Email: <your_email@example.com>

16. Acknowledgment

This repository includes implementations and experimental pipelines for hypergraph backbones, influence-function-based unlearning modules, tabular blind unlearning baselines, and privacy evaluation used in the TabHGIF project. We thank the open-source community for the libraries and tools used in this work.