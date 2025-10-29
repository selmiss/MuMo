"""
Script to upload a trained model to Hugging Face Hub.

Usage:
    python hub.py --model_path /path/to/model --repo_id username/model-name --token your_hf_token

Or set HF_TOKEN environment variable:
    export HF_TOKEN=your_token
    python hub.py --model_path /path/to/model --repo_id username/model-name
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoConfig, AutoTokenizer, AutoModel


def upload_model_to_hub(
    model_path: str,
    repo_id: str,
    token: str = None,
    private: bool = False,
    commit_message: str = "Upload model",
    create_model_card: bool = True,
):
    """
    Upload a trained model to Hugging Face Hub.
    
    Args:
        model_path: Path to the local model directory
        repo_id: Repository ID on Hugging Face Hub (e.g., "username/model-name")
        token: Hugging Face API token (or use HF_TOKEN env variable)
        private: Whether to create a private repository
        commit_message: Commit message for the upload
        create_model_card: Whether to create a basic model card
    """
    # Get token from environment if not provided
    # If None, huggingface_hub will use cached token from `huggingface-cli login`
    if token is None:
        token = os.environ.get("HF_TOKEN")
        # Don't raise error - let huggingface_hub handle it with cached credentials
    
    # Verify model path exists
    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    
    print(f"📦 Preparing to upload model from: {model_path}")
    print(f"🎯 Target repository: {repo_id}")
    print(f"🔒 Private: {private}")
    
    # Check if model files exist
    safetensor_files = list(model_path.glob("*.safetensors"))
    bin_files = list(model_path.glob("pytorch_model*.bin"))
    config_file = model_path / "config.json"
    
    if not config_file.exists():
        print("⚠️  Warning: config.json not found in model directory")
    
    if not safetensor_files and not bin_files:
        print("⚠️  Warning: No model weights found (.safetensors or .bin files)")
    else:
        if safetensor_files:
            print(f"✓ Found {len(safetensor_files)} safetensor file(s)")
        if bin_files:
            print(f"✓ Found {len(bin_files)} .bin file(s)")
    
    # Initialize Hugging Face API
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    try:
        print(f"\n🔨 Creating repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,  # Don't fail if repo already exists
            repo_type="model",
        )
        print(f"✓ Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠️  Repository creation note: {e}")
    
    # Create a basic model card if requested
    if create_model_card:
        readme_path = model_path / "README.md"
        if not readme_path.exists():
            print("\n📝 Creating basic model card (README.md)")
            model_card_content = f"""---
license: apache-2.0
tags:
- chemistry
- drug-discovery
- molecular-modeling
- mumo
---

# {repo_id.split('/')[-1]}

This model was trained using MuMo (Multi-Modal Molecular) framework.

## Model Description

- **Model Type**: MuMo Pretrained Model
- **Training Data**: Molecular structures and properties
- **Framework**: PyTorch + Transformers

## Usage

```python
from transformers import AutoConfig, AutoTokenizer, AutoModel

# Load model
model_path = "{repo_id}"
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

# Example usage
smiles = "CCO"  # Ethanol
inputs = tokenizer(smiles, return_tensors="pt")
outputs = model(**inputs)
```

## Training Details

- Training script: See repository for details
- Framework: Transformers + DeepSpeed

## Citation

If you use this model, please cite the original MuMo paper.

"""
            with open(readme_path, "w") as f:
                f.write(model_card_content)
            print("✓ Created README.md")
    
    # Upload the entire model folder
    print(f"\n🚀 Uploading model to Hugging Face Hub...")
    print(f"   This may take a few minutes depending on model size...")
    
    try:
        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            token=token,
            commit_message=commit_message,
            repo_type="model",
        )
        print(f"\n✅ Model successfully uploaded!")
        print(f"🔗 View your model at: https://huggingface.co/{repo_id}")
        print(f"\n📥 To download later, use:")
        print(f'   from transformers import AutoModel')
        print(f'   model = AutoModel.from_pretrained("{repo_id}", trust_remote_code=True)')
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Upload a trained model to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a public model
  python hub.py --model_path ./model/pretrain/mumo --repo_id username/mumo-pretrained

  # Upload a private model with custom message
  python hub.py --model_path ./model/pretrain/mumo --repo_id username/mumo-pretrained --private --commit_message "Initial upload"

  # Using environment variable for token
  export HF_TOKEN=hf_xxxxxxxxxxxxx
  python hub.py --model_path ./model/pretrain/mumo --repo_id username/mumo-pretrained
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the local model directory containing model files",
    )
    
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help='Repository ID on Hugging Face Hub (format: "username/model-name")',
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN environment variable)",
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository (default: public)",
    )
    
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload model",
        help="Commit message for the upload",
    )
    
    parser.add_argument(
        "--no_model_card",
        action="store_true",
        help="Don't create a model card (README.md) if it doesn't exist",
    )
    
    args = parser.parse_args()
    
    # Expand environment variables and user home directory
    model_path = os.path.expandvars(os.path.expanduser(args.model_path))
    
    try:
        upload_model_to_hub(
            model_path=model_path,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
            commit_message=args.commit_message,
            create_model_card=not args.no_model_card,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

