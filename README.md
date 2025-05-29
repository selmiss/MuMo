# **MuMo: Multimodal Molecular Representation Learning via Structural Fusion and Progressive Injection (Anonymous Submission)**

*(Anonymous Submission for NeurIPS 2025, Paper ID: 17946)*

This repository contains the code, dataset, and trained models used in our research paper submitted to a leading AI conference. Due to the double-blind review process, all authorship details have been removed.

**Follow the instructions below, we believe you can reproduce the whole pretrain and finetune results within 24h using 4*A100-80G GPUs.**

**Abstract**

Multimodal molecular models often suffer from 3D conformer sensitivity and modality mismatch, limiting their robustness and generalization. We propose \textbf{MuMo}, a structured fusion framework that addresses these challenges through two key components. To reduce the unstable of conformer-dependent fusion, we design a structured fusion pipeline (SFP) that combines 2D topology and 3D geometry into a stable structural prior. To mitigate modality mismatch from symmetric fusion, we introduce an Injection-Enhanced Attention (IEA) mechanism that asymmetrically integrates this prior into the sequence stream, preserving modality-specific modeling while enabling cross-modal enrichment. 
Built on a state space backbone, MuMo supports long-range dependency modeling and robust information propagation. 
Across 22 benchmark tasks from TDC and MoleculeNet, MuMo achieves an average improvement of 2.7\% over the best-performing baseline on each task, ranking first on 17 of them, including a 27\% reduction in MAE on LD50. These results validate its robustness to 3D conformer noise and the effectiveness of asymmetric, structure-aware fusion.

![structure](./fig/structure.png)
![structure](./fig/fusion.jpg)

---

## **1. Repository Overview**

This repository provides:

- **Code**: Implementation of the proposed method
- **Dataset**: Raw datasets and preprocessed dataset for pretraining and finetuning
- **Trained Models**: Pre-trained model files
- **Reproducibility**: Instructions for setting up the environment and running

---

## **2. Dependencies & Environment**

To ensure reproducibility, we provide a list of required dependencies.

### **Using Conda (Recommended)**

```bash
# Create and activate conda environment
conda create -n mumo python=3.11
conda activate mumo

# Install PyTorch with CUDA support
conda install pytorch==2.5.1 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install other requirements
pip install -r requirements.txt

# If you want to get the traning and inference speed from mamba structure, you would like to install:
# Install causal-conv1d from source
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
pip install -e .
cd ..

# Install mamba from source
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install -e .
cd ..
```

### **Using pip**

```bash
# Install PyTorch with CUDA support
pip install torch==2.5.1+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other requirements
pip install -r requirements.txt

# If you want to get the traning and inference speed from mamba structure, you would like to install:
# Install causal-conv1d from source
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
pip install -e .
cd ..

# Install mamba from source
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install -e .
cd ..
```

Note: Make sure you have CUDA 12.4 installed on your system. If you're using a different CUDA version, you'll need to adjust the PyTorch installation command accordingly.

## Installation

1. First, install the required packages:
```bash
pip install -r requirements.txt
```

2. Install causal-conv1d from source:
```bash
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
pip install -e .
```

3. Install mamba from source:
```bash
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install -e .
```

---

## **3. Dataset**

The pretrain and finetune dataset s used in our experiments can be downloaded from the anonymous storage link:

🔗 [Finetune Dataset (Anonymous)](https://drive.google.com/file/d/1-KVM21Hc1pdx4p3agxqiuIuk-Gur_5KO/view?usp=sharing) 243.8M

🔗 [Pretrain Dataset (Anonymous)](https://drive.google.com/file/d/16m476wsvnVVbo6fD5qNVAeVN4FLathX-/view?usp=sharing) 1.66G

After downloading the ZIP compressed dataset, extract it to obtain the `dataset` folder. If both pretraining and fine-tuning datasets are downloaded, merge them into a single `dataset` folder.

Place the extracted and merged `dataset` folder in an appropriate location on your system and remember its path. Then, update the `DATA_DIR` variable in the bash scripts you need to run within this repository to point to the dataset folder's path. 

Similarly, update the `BASE_DIR` variable to the path where you have placed the `MuMo` repository (including the `MuMo` folder itself). 

Ensure that **neither `DATA_DIR` nor `BASE_DIR` includes a trailing slash in the end**. It is recommended to use VS Code's search-and-replace feature for quick modifications, search and replace `DATA_DIR` and `BASE_DIR`.

If you want to process your own data, please use `mol3d_processor.py` to do it, there is an example in it.
---

## **4. Model Checkpoints**

Pre-trained models are available at:

🔗 [Pretrained Model (Anonymous)](https://drive.google.com/file/d/1J5vNYV9q7rqpVIZsFuuqU6CrBqys7K2P/view?usp=sharing) 1.82G

After downloading and extracting the pre-trained model, you will obtain a folder named `model`. Place this `model` folder in the same directory as the `dataset` folder.

---

## **5. Running the Code**

Make sure you put the data and model in the right place and update the `DATA_DIR`and `BASE_DIR`  in `.sh` files with the right address.

We run pretraining on 4\*A100-80GPUs, and finetuning on 2\*A100-80GPUs using deepspeed architecture.  So, it's recommended you have 4 A100-80G GPUs. If not, please adjust the batch size to fit your computing hardware.

### (0) One Line Run

**One line of code to run all the pipeline! (Pretrain & Finetune)**

```bash
cd MuMo/scripts/; bash reproduce.sh
```

Be sure that you prepare pretrain datasets and finetuning datasets in a `dataset`directory.

### **(1) Pretrain from scratch**

First, you should download the datasets (link is provided above) to your data directory.

To train the model from scratch, use:

```bash
cd MuMo
bash ./scripts/pretrain/mumo.sh
```

### **(2) Finetuning**

To finetune the pretrained model, run:

```bash
cd MuMo
master_port=29500 gpus=0,1 bash ./scripts/sft/tuning_all.sh mumo bace,bbbp,clintox,tox21,sider,delaney,lipo,freesolv MuMoFinetune sft_geo_randomsplit;
master_port=29500 gpus=0,1 bash ./scripts/sft/tuning_all.sh mumo bace,bbbp MuMoFinetune sft_geo_scaffoldsplit;
```

When the training finished, the results will be written to `./results/`. We have already  provided the results we trained for the paper in the `./results` folder.

## 6. Results

![ ](./fig/results.png)

Note that different GPUs or batch size may influence the results slightly, but overall the same.

## **7. Citation**

To ensure double-blind review, citation details will be provided upon paper acceptance.

---

## **8. Contact**

Since this is an anonymous submission, we cannot provide direct contact details. Any questions should be addressed through the conference review system.
