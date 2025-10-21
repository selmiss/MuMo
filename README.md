# **MuMo: Multimodal Molecular Representation Learning via Structural Fusion and Progressive Injection**

*Accepted at NeurIPS 2025*

Authors: Zihao Jing1, Yan Sun1, Yan Yi Li2, Sugitha Janarthanan2, Alana Deng1, Pingzhao Hu1,2∗

1 Department of Computer Science, Western University, London, ON, Canada

2 Department of Biochemistry, Western University, London, ON, Canada

Contact: zjing29@uwo.ca, phu49@uwo.ca

This repository contains the code, dataset, and trained models accompanying our NeurIPS 2025 paper.

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

## **3. Dataset**

The pretrain and finetune datasets used in our experiments can be downloaded from the following links:

🔗 [Finetune Dataset](https://drive.google.com/file/d/1-KVM21Hc1pdx4p3agxqiuIuk-Gur_5KO/view?usp=sharing) 243.8M

🔗 [Pretrain Dataset](https://drive.google.com/file/d/16m476wsvnVVbo6fD5qNVAeVN4FLathX-/view?usp=sharing) 1.66G

After downloading the ZIP compressed dataset, extract it to obtain the `dataset` folder. If both pretraining and fine-tuning datasets are downloaded, merge them into a single `dataset` folder.

Place the extracted and merged `dataset` folder in an appropriate location on your system and update the `DATA_DIR` variable in `init_env.sh` to point to the directory containing the dataset folder (see **Environment Setup** section above).

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

![ ](./fig/result.png)

Note that different GPUs or batch size may influence the results slightly, but overall the same.

## **7. Citation**

If you find this work useful, please cite:

Jing, Zihao; Sun, Yan; Li, Yan Yi; Janarthanan, Sugitha; Deng, Alana; Hu, Pingzhao. "MuMo: Multimodal Molecular Representation Learning via Structural Fusion and Progressive Injection." In Advances in Neural Information Processing Systems (NeurIPS), 2025.

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
