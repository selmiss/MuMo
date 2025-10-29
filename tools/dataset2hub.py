import os
from pathlib import Path
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi, login

# Login to Hugging Face Hub
# Option 1: Use token from environment variable
# login(token=os.environ.get("HF_TOKEN"))

# Option 2: Login interactively (will prompt for token)
# login()

# Option 3: If already logged in via CLI, no need to call login()
# The token from ~/.cache/huggingface/token will be used automatically

# Step 1: Load your local JSONL files into a Dataset
def upload_dataset_from_local():
    dataset = load_dataset(
        "json",
        data_files={
            "train": "/data/lab_ph/zihao/Nips/dataset/pretrain/chembl_train_dict.jsonl",
            "validation": "/data/lab_ph/zihao/Nips/dataset/pretrain/chembl_eval_dict.jsonl"  # optional
        }
    )

    # Step 2: Push to Hugging Face Hub
    dataset.push_to_hub(
        "zihaojing/MuMo-Pretraining",  # Replace with your HF username and desired dataset name
        private=False  # Set to True if you want a private dataset
    )


def upload_multiple_finetune_datasets(
    base_dir,
    repo_name,
    task_folders=None,
    train_file="train.jsonl",
    valid_file="valid.jsonl",
    test_file="test.jsonl",
    private=False
):
    """
    Upload multiple finetune datasets to a single Hugging Face Hub repository.
    Each task will be uploaded as a separate configuration (subset).
    
    Args:
        base_dir (str): Base directory containing task folders
        repo_name (str): Hugging Face Hub repo name (e.g., "username/finetune-datasets")
        task_folders (list, optional): List of task folder names. If None, auto-detect all folders.
        train_file (str): Name of training file in each task folder
        valid_file (str): Name of validation file in each task folder
        test_file (str): Name of test file in each task folder
        private (bool): Whether to make the repository private
    
    Example directory structure:
        base_dir/
        ├── AMES/
        │   ├── train.jsonl
        │   ├── valid.jsonl
        │   └── test.jsonl
        ├── BBBP/
        │   ├── train.jsonl
        │   ├── valid.jsonl
        │   └── test.jsonl
        └── ...
    
    After uploading, load specific tasks like:
        dataset = load_dataset("username/finetune-datasets", "AMES")
        dataset = load_dataset("username/finetune-datasets", "BBBP")
    """
    base_path = Path(base_dir)
    
    # Auto-detect task folders if not specified
    if task_folders is None:
        task_folders = [d.name for d in base_path.iterdir() if d.is_dir()]
        print(f"Auto-detected {len(task_folders)} tasks: {task_folders}")
    
    # Upload each task as a separate configuration
    for task_name in task_folders:
        task_path = base_path / task_name
        
        if not task_path.exists():
            print(f"Warning: Task folder '{task_name}' not found, skipping...")
            continue
        
        # Build data files dict
        data_files = {}
        
        train_path = task_path / train_file
        valid_path = task_path / valid_file
        test_path = task_path / test_file
        
        if train_path.exists():
            data_files["train"] = str(train_path)
        if valid_path.exists():
            data_files["validation"] = str(valid_path)
        if test_path.exists():
            data_files["test"] = str(test_path)
        
        if not data_files:
            print(f"Warning: No data files found for task '{task_name}', skipping...")
            continue
        
        print(f"\nUploading task '{task_name}' with splits: {list(data_files.keys())}")
        
        # Load dataset
        dataset = load_dataset("json", data_files=data_files)
        
        # Push to hub with config name
        dataset.push_to_hub(
            repo_name,
            config_name=task_name,
            private=private
        )
        
        print(f"✓ Successfully uploaded task '{task_name}'")
    
    print(f"\n{'='*60}")
    print(f"All tasks uploaded to: https://huggingface.co/datasets/{repo_name}")
    print(f"{'='*60}")
    print("\nTo load a specific task:")
    print(f'  dataset = load_dataset("{repo_name}", "{task_folders[0]}")')


def upload_finetune_with_structure(
    base_dir,
    repo_name,
    private=False
):
    """
    Alternative approach: Upload with folder structure preserved.
    This uploads files directly maintaining the directory structure.
    
    Args:
        base_dir (str): Base directory containing task folders
        repo_name (str): Hugging Face Hub repo name
        private (bool): Whether to make the repository private
    
    After uploading, load like:
        dataset = load_dataset("username/finetune-datasets", data_files={
            "train": "AMES/train.jsonl",
            "validation": "AMES/valid.jsonl"
        })
    """
    api = HfApi()
    
    # Create the repository if it doesn't exist
    try:
        api.create_repo(repo_id=repo_name, repo_type="dataset", private=private, exist_ok=True)
        print(f"Repository '{repo_name}' ready")
    except Exception as e:
        print(f"Note: {e}")
    
    base_path = Path(base_dir)
    
    # Upload all JSONL files
    for jsonl_file in base_path.rglob("*.jsonl"):
        relative_path = jsonl_file.relative_to(base_path)
        
        print(f"Uploading: {relative_path}")
        
        api.upload_file(
            path_or_fileobj=str(jsonl_file),
            path_in_repo=str(relative_path),
            repo_id=repo_name,
            repo_type="dataset",
        )
    
    print(f"\n{'='*60}")
    print(f"All files uploaded to: https://huggingface.co/datasets/{repo_name}")
    print(f"{'='*60}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Upload pretraining dataset
    # upload_dataset_from_local()
    
    # Example 2: Upload multiple finetune datasets (RECOMMENDED)
    # This creates separate configs for each task
    # """
    upload_multiple_finetune_datasets(
        base_dir="/data/lab_ph/zihao/Nips/hub_upload",
        repo_name="zihaojing/MuMo-Finetune",
        private=False  # Set to True for private dataset
    )
    
    # Then load specific tasks like:
    # dataset = load_dataset("zihaojing/MuMo-Finetune", "AMES")
    # dataset = load_dataset("zihaojing/MuMo-Finetune", "BBBP")
    # """
    
    # Example 3: Upload with folder structure preserved (ALTERNATIVE)
    """
    upload_finetune_with_structure(
        base_dir="/data/lab_ph/zihao/Nips/dataset/finetune",
        repo_name="zihaojing/MuMo-Finetune",
        private=False
    )
    
    # Then load like:
    # dataset = load_dataset("zihaojing/MuMo-Finetune", 
    #                        data_files={"train": "AMES/train.jsonl", 
    #                                    "validation": "AMES/valid.jsonl"})
    """
    
    # Example 4: Upload specific tasks only
    """
    upload_multiple_finetune_datasets(
        base_dir="/data/lab_ph/zihao/Nips/dataset/finetune",
        repo_name="zihaojing/MuMo-Finetune",
        task_folders=["AMES", "BBBP", "ClinTox"],  # Only these tasks
        private=False
    )
    """
    
    # print("Please uncomment the example you want to run in the __main__ section.")