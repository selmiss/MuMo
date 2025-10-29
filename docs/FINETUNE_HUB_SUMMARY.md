# Finetune Scripts - Hub Dataset Support

## Summary of Changes

All three finetuning scripts have been updated to support loading datasets from Hugging Face Hub while retaining the original local file loading functionality.

### Updated Files

1. ✅ `train/finetune.py`
2. ✅ `train/finetune_reaction.py`
3. ✅ `train/pairwise_sft.py`

### How It Works

The scripts now check if `--dataset_name` is provided:
- **If yes**: Load from Hugging Face Hub
- **If no**: Use existing local file loading (no breaking changes)

## Usage Examples

### Load from Hugging Face Hub (NEW)

```bash
# Load AMES task from Hub
python train/finetune.py \
    --dataset_name zihaojing/MuMo-Finetune \
    --dataset_config_name AMES \
    --model_class MuMoFinetune \
    --task_type classification \
    --output_size 2 \
    --model_name_or_path /path/to/pretrained/model \
    --tokenizer_name /path/to/tokenizer \
    --output_dir ./output/AMES \
    --do_train \
    --do_eval

# Load BBBP task - just change config name!
python train/finetune.py \
    --dataset_name zihaojing/MuMo-Finetune \
    --dataset_config_name BBBP \
    # ... rest same as above
```

### Load from Local Files (ORIGINAL - Still Works)

```bash
# Original method - no changes needed
python train/finetune.py \
    --train_files /path/to/AMES/train.jsonl \
    --validation_files /path/to/AMES/valid.jsonl \
    --test_files /path/to/AMES/test.jsonl \
    --model_class MuMoFinetune \
    # ... other arguments
```

## Workflow: Upload & Use

### Step 1: Upload Your Finetune Datasets

Edit `tools/hub.py` and run:

```python
# In tools/hub.py - uncomment and modify:
upload_multiple_finetune_datasets(
    base_dir="/path/to/your/finetune/datasets",
    repo_name="username/YourRepo-Name",
    private=False
)
```

```bash
python tools/hub.py
```

This uploads all tasks (AMES, BBBP, ClinTox, etc.) as separate configs.

### Step 2: Use in Training

```bash
# Train on any task by specifying the config name
python train/finetune.py \
    --dataset_name username/YourRepo-Name \
    --dataset_config_name AMES \
    # ... other args
```

## Key Benefits

### 1. Easy Task Switching
```bash
# Just change one parameter to switch tasks
--dataset_config_name AMES
--dataset_config_name BBBP
--dataset_config_name ClinTox
```

### 2. No Path Management
Before:
```bash
--train_files /long/path/to/dataset/AMES/train.jsonl \
--validation_files /long/path/to/dataset/AMES/valid.jsonl \
--test_files /long/path/to/dataset/AMES/test.jsonl \
```

After:
```bash
--dataset_name zihaojing/MuMo-Finetune \
--dataset_config_name AMES \
```

### 3. Consistent Across Team
Everyone loads the exact same dataset version from the Hub - no file sync issues.

### 4. Automatic Splits
Hub datasets come with train/validation/test splits already defined.

## What's Supported

### All File Formats (Local Loading)
- ✅ JSONL files (recommended for Hub)
- ✅ CSV files
- ✅ TXT files

### Hub Loading
- ✅ Public datasets (no authentication needed)
- ✅ Private datasets (with `--use_auth_token` flag)
- ✅ Streaming mode (`--streaming`)
- ✅ All three splits (train/validation/test)

## Parameters Reference

### For Hub Loading
- `--dataset_name`: Hub dataset ID (e.g., `username/dataset-name`)
- `--dataset_config_name`: Task name (e.g., `AMES`, `BBBP`)
- `--use_auth_token`: (Optional) For private datasets
- `--streaming`: (Optional) Enable streaming mode

### For Local Loading (Original)
- `--train_files`: Path(s) to training file(s)
- `--validation_files`: Path(s) to validation file(s)
- `--test_files`: Path(s) to test file(s)

## No Breaking Changes

✅ All existing scripts and workflows continue to work exactly as before.
✅ Local file loading is unchanged.
✅ Only new capability added: Hub loading.

## Examples for Each Script

### finetune.py
```bash
# Classification task from Hub
python train/finetune.py \
    --dataset_name zihaojing/MuMo-Finetune \
    --dataset_config_name AMES \
    --task_type classification \
    --output_size 2
```

### finetune_reaction.py
```bash
# Reaction task from Hub
python train/finetune_reaction.py \
    --dataset_name zihaojing/MuMo-Finetune \
    --dataset_config_name USPTO \
    --task_type regression
```

### pairwise_sft.py
```bash
# Pairwise training from Hub
python train/pairwise_sft.py \
    --dataset_name zihaojing/MuMo-Finetune \
    --dataset_config_name PairwiseData \
    --task_type regression
```

## Authentication

### Public Datasets (like zihaojing/MuMo-Finetune)
**No authentication needed!** Just use `--dataset_name`.

### Private Datasets
1. Login once: `huggingface-cli login`
2. Add flag: `--use_auth_token`

## Documentation

See `DATASET_LOADING_GUIDE.md` for complete documentation on:
- How to upload datasets
- How to load datasets
- Authentication details
- Troubleshooting

See `tools/README.md` for details on uploading multiple finetune datasets.

## Quick Start

1. **Upload your datasets:**
   ```bash
   python tools/hub.py  # After configuring base_dir and repo_name
   ```

2. **Train on any task:**
   ```bash
   python train/finetune.py \
       --dataset_name zihaojing/MuMo-Finetune \
       --dataset_config_name AMES \
       # ... other training args
   ```

3. **Switch tasks easily:**
   Just change `--dataset_config_name` to another task name!

## Verified Working

✅ No linter errors in any of the three updated scripts
✅ Backward compatible with existing local file loading
✅ Consistent with `pretrain.py` implementation

