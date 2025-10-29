# Dataset Loading Guide

This guide explains how to load datasets for MuMo (pretraining and finetuning) from either local files or Hugging Face Hub.

## Option 1: Load from Local Files (Default)

Use the original `scripts/pretrain/mumo.sh` script with:

```bash
--train_files ${DATA_DIR}/dataset/pretrain/chembl_train_dict.jsonl \
--validation_files ${DATA_DIR}/dataset/pretrain/chembl_eval_dict.jsonl \
```

**Example:**
```bash
bash scripts/pretrain/mumo.sh
```

## Option 2: Load from Hugging Face Hub

### Step 1: Login to Hugging Face (Only needed for uploading or private datasets)

**For PUBLIC datasets:** No login needed! Skip to Step 3.

**For uploading datasets or accessing private datasets:**

```bash
huggingface-cli login
```

You'll need to get a token from: https://huggingface.co/settings/tokens
- Make sure the token has "write" permissions if you plan to upload datasets

### Step 2: Upload Your Dataset (If not already uploaded)

Use the `tools/hub.py` script to upload your JSONL files:

```python
# tools/hub.py already configured for:
# - Train: /data/lab_ph/zihao/Nips/dataset/pretrain/chembl_train_dict.jsonl
# - Validation: /data/lab_ph/zihao/Nips/dataset/pretrain/chembl_eval_dict.jsonl
# - Destination: zihaojing/MuMo-Pretraining

python tools/hub.py
```

### Step 3: Use the Hub Dataset in Training

**Method A:** Use the provided `mumo_hub.sh` script:
```bash
bash scripts/pretrain/mumo_hub.sh
```

**Method B:** Modify your existing script to use Hub loading:

Replace:
```bash
--train_files ${DATA_DIR}/dataset/pretrain/chembl_train_dict.jsonl \
--validation_files ${DATA_DIR}/dataset/pretrain/chembl_eval_dict.jsonl \
```

With:
```bash
--dataset_name zihaojing/MuMo-Pretraining \
# Note: --use_auth_token is NOT needed for public datasets
```

## Key Parameters

### For Local File Loading:
- `--train_files`: Path(s) to training file(s)
- `--validation_files`: Path(s) to validation file(s)

### For Hub Loading:
- `--dataset_name`: Hugging Face Hub dataset identifier (format: `username/dataset-name`)
- `--dataset_config_name`: (Optional) Dataset configuration name
- `--use_auth_token`: **ONLY needed for private/gated datasets** (NOT needed for public datasets like zihaojing/MuMo-Pretraining)

## Advantages of Using Hugging Face Hub

1. **Version Control**: Track dataset versions
2. **Easy Sharing**: Share datasets with collaborators
3. **Automatic Caching**: Datasets are cached locally after first download
4. **Streaming Support**: Load large datasets without downloading everything
5. **Dataset Cards**: Document your dataset with metadata
6. **Reduced Storage**: No need to keep multiple copies on different machines

## Finetuning with Hub Datasets

All finetuning scripts now support loading from Hugging Face Hub:
- `train/finetune.py`
- `train/finetune_reaction.py`
- `train/pairwise_sft.py`

### Loading Specific Tasks from Hub

After uploading your finetune datasets using `tools/hub.py`, you can load specific tasks:

```bash
python train/finetune.py \
    --dataset_name zihaojing/MuMo-Finetune \
    --dataset_config_name AMES \
    --model_class MuMoFinetune \
    --task_type classification \
    --output_size 2 \
    # ... other arguments
```

### Example: AMES Classification Task

```bash
# From Hub
--dataset_name zihaojing/MuMo-Finetune \
--dataset_config_name AMES \

# From local files (original method)
--train_files /path/to/AMES/train.jsonl \
--validation_files /path/to/AMES/valid.jsonl \
--test_files /path/to/AMES/test.jsonl \
```

### Benefits for Finetuning

1. **Easy Task Switching**: Just change `--dataset_config_name` to switch between tasks
2. **No Path Management**: No need to manage file paths for each task
3. **Consistent Data**: Ensure all team members use the same dataset version
4. **Quick Experiments**: Easily run experiments across multiple tasks

## Notes

- JSONL (JSON Lines) format is fully supported and recommended
- When loading from Hub, the code assumes the dataset is already in JSONL format
- Both methods support streaming mode with `--streaming` flag
- The validation split is automatically detected from the Hub dataset if present
- All finetuning scripts (`finetune.py`, `finetune_reaction.py`, `pairwise_sft.py`) support Hub loading

## Authentication: When Do You Need It?

### ✅ You DON'T need authentication (no login required) for:
- **Public datasets** like `zihaojing/MuMo-Pretraining`
- Public models
- Just downloading/using public resources

### ⚠️ You NEED authentication (login required) for:
- **Uploading datasets** to Hugging Face Hub
- Accessing **private datasets**
- Accessing **gated datasets** (require approval)
- Using **private models**

**To login:**
```bash
huggingface-cli login
```

## Troubleshooting

### Authentication Errors
If you see authentication errors when using a **public** dataset, you likely don't need authentication at all. Just remove the `--use_auth_token` flag.

For **private/gated** resources:
```bash
huggingface-cli whoami  # Check if logged in
huggingface-cli login   # Login if needed
```

### Dataset Not Found
Make sure:
1. The dataset name is correct (format: `username/dataset-name`)
2. You have access permissions (for private datasets)
3. The `--use_auth_token` flag is included

### Memory Issues with Large Datasets
Use streaming mode:
```bash
--streaming \
```

This loads data on-the-fly without loading the entire dataset into memory.

