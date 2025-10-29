# **MuMo: Multimodal Molecular Representation Learning via Structural Fusion and Progressive Injection**

*Accepted at NeurIPS 2025*

- 📄 **Paper**: [NeurIPS 2025 Poster](https://neurips.cc/virtual/2025/poster/119127)
- 🧠 **Pretrained Model**: [zihaojing/MuMo-Pretrained](https://huggingface.co/zihaojing/MuMo-Pretrained)
- 🗂️ **Datasets**: [zihaojing/MuMo-Finetuning](https://huggingface.co/datasets/zihaojing/MuMo-Finetuning) · [zihaojing/MuMo-Pretraining](https://huggingface.co/datasets/zihaojing/MuMo-Pretraining)

Authors: Zihao Jing¹, Yan Sun¹, Yanyi Li², Sugitha Janarthanan², Alana Deng¹, Pingzhao Hu¹²∗

1 Department of Computer Science, Western University, London, ON, Canada

2 Department of Biochemistry, Western University, London, ON, Canada

Contact: 

Zihao Jing: zjing29@uwo.ca | Wechat: A2016A315214 | Instagram: nobeljing25

Pingzhao Hu: phu49@uwo.ca

This repository contains the code, datasets, and trained models accompanying our NeurIPS 2025 paper.

**Follow the instructions below, we believe you can reproduce the whole pretrain and finetune results within 24h using 4*A100-80G GPUs.**

**Abstract**

Multimodal molecular models often suffer from 3D conformer sensitivity and modality mismatch, limiting their robustness and generalization. We propose \textbf{MuMo}, a structured fusion framework that addresses these challenges through two key components. To reduce the unstable of conformer-dependent fusion, we design a structured fusion pipeline (SFP) that combines 2D topology and 3D geometry into a stable structural prior. To mitigate modality mismatch from symmetric fusion, we introduce an Injection-Enhanced Attention (IEA) mechanism that asymmetrically integrates this prior into the sequence stream, preserving modality-specific modeling while enabling cross-modal enrichment. 
Built on a state space backbone, MuMo supports long-range dependency modeling and robust information propagation. 
Across 22 benchmark tasks from TDC and MoleculeNet, MuMo achieves an average improvement of 2.7\% over the best-performing baseline on each task, ranking first on 17 of them, including a 27\% reduction in MAE on LD50. These results validate its robustness to 3D conformer noise and the effectiveness of asymmetric, structure-aware fusion.

![structure](./fig/structure.png)
![structure](./fig/fusion.jpg)

---

## **1. Repository Overview**

MuMo is a structured multimodal molecular learning framework that fuses 2D topology and 3D geometry with sequence signals using:

- **Structured Fusion Pipeline (SFP)**: builds a stable structural prior by combining 2D graphs and 3D conformers
- **Progressive Injection (PI)**: asymmetrically injects the prior into the token stream to avoid modality collapse
- **State space backbone (Mamba)**: supports long-range dependencies and efficient training/inference

This repo provides:

- **Code**: Pretraining, finetuning (TDC, MoleculeNet, QM), and inference
- **Datasets**: Pretraining and finetuning datasets on the Hugging Face Hub
- **Checkpoints**: Pretrained model on the Hub for downstream use
- **Reproducibility**: Scripts to reproduce results with DeepSpeed

---

## **2. Dependencies & Environment**

To ensure reproducibility, we provide a list of required dependencies.

### **Using Conda (Recommended)**

Be careful about the cuda-toolkit version in the environment.yml, it should fit your cuda driver's version (not higher than the nvidia driver). And also, your cuda-toolkit version should be consistent as the cuda version you compile the pytorch.
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate mumo

# If you want to get the traning and inference speed from mamba structure, you would like to install:
# Install causal-conv1d from source
git clone https://github.com/Dao-AILab/causal-conv1d.git
pip install ./causal-conv1d

# Install mamba-ssm from source
git clone https://github.com/state-spaces/mamba.git
pip install ./mamba

# make sure your transformer version is compatible.
conda install transformers==4.45.2
conda install pytorch==2.5.1
conda install numpy~=2.2
```

---

## **Environment Setup**

Before running any scripts, you must configure the environment variables for your project and data directories.

1. **Edit the environment configuration file:**
   ```bash
   # Open init_env.sh and update the paths
   nano init_env.sh  # or use your preferred editor
   ```

2. **Update the paths in `init_env.sh`:**
   ```bash
   export BASE_DIR=/path/to/your/MuMo  # Change to your project directory path
   export DATA_DIR=/path/to/your/data  # Change to your data directory path
   export PYTHONPATH=${BASE_DIR}
   ```

3. **Source the environment file before running any scripts:**
   ```bash
   source init_env.sh
   ```

**Important Notes:**
- Ensure that **neither `DATA_DIR` nor `BASE_DIR` includes a trailing slash at the end**
- `BASE_DIR` should point to the root directory of this MuMo repository
- `DATA_DIR` should point to the directory containing your dataset folder
- You must run `source init_env.sh` in every new terminal session before executing any scripts
- All training, inference, and preprocessing scripts depend on these environment variables

---

## **3. Datasets on the Hub**

All datasets used in the paper are hosted on the Hugging Face Hub:

- 🗂️ Finetuning datasets (TDC, QM, Reaction Yield, etc.): [`zihaojing/MuMo-Finetuning`](https://huggingface.co/datasets/zihaojing/MuMo-Finetuning)
  - Includes tasks such as AMES, BBBP, DILI, LD50_Zhu, Lipophilicity, etc.
  - Already processed with graph and geometry features `x`, `edge_index`, `edge_attr`; includes SMILES as `smiles`
  - Splits: `train/validation/test` per task

- 🗂️ Pretraining dataset: [`zihaojing/MuMo-Pretraining`](https://huggingface.co/datasets/zihaojing/MuMo-Pretraining)
  - Large-scale corpus of molecular SMILES for pretraining
  - Used with on-the-fly graph construction in the training pipeline

You can point scripts directly to these Hub datasets via `--dataset_name` and `--dataset_config_name ${TASK_NAME}` (no local files needed). If you prefer local files, see `preprocess/mol3d_processor.py` for data processing utilities.

### 3.1 Local dataset layout (only for custom/local files)

Folder names must be consistent with the `DATATYPE` and `TASK_NAME` used in scripts (e.g., `scripts/sft_tdc/regression/LD50.sh`, `scripts/sft_QM/qm7.sh`).

```text
${DATA_DIR}/dataset/
  └── ${DATATYPE}/
      └── ${TASK_NAME}/
          ├── train.jsonl
          ├── valid.jsonl
          └── test.jsonl
```

- Examples:
  - `DATATYPE=tdc_geo_tox`, `TASK_NAME=LD50_Zhu`
  - `DATATYPE=QM`, `TASK_NAME=qm7`

Use the same `TASK_NAME` string in your scripts and folder name to avoid mismatches.

### 3.2 File formats and schema (before vs. after processing)

- Before processing: CSV files

```csv
smiles,Y
CCO,1
CC(=O)O,0
```

  - Columns:
    - `smiles`: SMILES string
    - `Y`: label column (classification/regression). For QM tasks, the label name can differ (e.g., `u0_atom` in QM7). Set via `--label_column_name`.

- After processing: JSONL files with graph features

```json
{"smiles": "CCO", "x": [[...]], "edge_index": [[...],[...]], "edge_attr": [[...]], "Y": 1}
```

  - Required graph keys:
    - `x`: node feature matrix (list of lists)
    - `edge_index`: 2×E edge indices (list of two lists)
    - `edge_attr`: edge feature matrix (list of lists)
  - Other optional keys are supported (e.g., geometry variants) but not required.

Script flags to bind columns:
- `--data_column_name smiles`
- `--label_column_name Y` (or your label, e.g., `u0_atom` for QM7)

### 3.3 Generate graph fields from SMILES

You can generate graph fields on the fly using `preprocess/mol3d_processor.py` or during training (see pretraining pipeline). Minimal example:

```python
from preprocess.mol3d_processor import smiles2GeoGraph

smiles = "CCO"
g = smiles2GeoGraph(smiles, brics=False, geo_operation=False)

record = {
    "smiles": smiles,
    "x": g.x.tolist(),
    "edge_index": g.edge_index.tolist(),
    "edge_attr": g.edge_attr.tolist(),
    # add your label(s) here, e.g., "Y": 1
}
```

Notes:
- For local finetuning, provide `train.jsonl`, `valid.jsonl`, `test.jsonl` under the task folder.
- Ensure `DATATYPE`/`TASK_NAME` match the script paths you run.

---

## **4. Model Checkpoints**

- 🧠 Pretrained checkpoint: [`zihaojing/MuMo-Pretrained`](https://huggingface.co/zihaojing/MuMo-Pretrained)

Load directly via Transformers:

```python
from transformers import AutoModel, AutoConfig, AutoTokenizer

repo = "zihaojing/MuMo-Pretrained"
config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModel.from_pretrained(repo, trust_remote_code=True)
```

---

## **5. Running the Code**

Make sure you set `BASE_DIR` and `DATA_DIR` in `init_env.sh` and source it before running. We pretrain on 4×A100-80G and finetune on 2×A100-80G with DeepSpeed. Adjust batch sizes if you have fewer resources.

### **(1) Pretrain from scratch**

First, you should download the datasets (link is provided above) to your data directory.

To train the model from scratch, use:

```bash
cd MuMo
bash ./scripts/pretrain/mumo.sh
```

### **(2) Finetuning Examples**

Below are minimal example scripts for common settings. Feel free to adjust GPU selection, ports, and batch sizes.

- TDC Classification (AMES): `scripts/sft_tdc/classfication/AMES.sh`

```bash
bash scripts/sft_tdc/classfication/AMES.sh
```

- TDC Regression (LD50"): `scripts/sft_tdc/regression/LD50.sh`

```bash
bash scripts/sft_tdc/regression/LD50.sh
```

- QM Task (QM7): `scripts/sft_QM/qm7.sh`

```bash
bash scripts/sft_QM/qm7.sh
```

### **(3) Inference Example**

For batch inference on IC50 with a finetuned checkpoint, use:

```bash
bash scripts/infer/infer_ic50.sh
```

## 6. Results

![ ](./fig/result.png)

Note that different GPUs or batch size may influence the results slightly, but overall the same.

## **7. Citation**

If you find this work useful, please cite:

Zihao Jing, Yan Sun, Yanyi Li, Sugitha Janarthanan, Alana Deng, and Pingzhao Hu. "MuMo: Multimodal Molecular Representation Learning via Structural Fusion and Progressive Injection." In Advances in Neural Information Processing Systems (NeurIPS), 2025. ([paper](https://neurips.cc/virtual/2025/poster/119127))

```bibtex
@inproceedings{jing2025mumo,
  title        = {MuMo: Multimodal Molecular Representation Learning via Structural Fusion and Progressive Injection},
  author       = {Jing, Zihao and Sun, Yan and Li, Yan Yi and Janarthanan, Sugitha and Deng, Alana and Hu, Pingzhao},
  booktitle    = {Advances in Neural Information Processing Systems (NeurIPS)},
  year         = {2025}
}
```

---

## **8. Contact**

For questions or collaboration, please contact: zjing29@uwo.ca (Zihao Jing) or phu49@uwo.ca (Pingzhao Hu, corresponding author).
