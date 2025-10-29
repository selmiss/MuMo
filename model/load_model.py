import torch.nn as nn
from transformers import AutoConfig
from model.attention_mamba import *
from model.mumo import *


def initialize_weights(model):
    """
    Init all the params in model.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def load_model(config, tokenizer=None, training_args=None, model_args=None):
    """
    Load Model using config and model_args

    Supports loading from:
    - Local path: "/path/to/model"
    - Hugging Face Hub: "username/model-name"
    - Training from scratch: set model_args.model_name_or_path to None

    Args:
        config: Model configuration
        tokenizer (optional): Tokenizer instance
        training_args (optional): Training arguments
        model_args (optional): Model arguments including model_name_or_path

    Raises:
        ValueError: If model class not found or invalid configuration

    Returns:
        model: Loaded model instance

    Examples:
        # Load from local path
        model_args.model_name_or_path = "/path/to/model"

        # Load from Hugging Face Hub
        model_args.model_name_or_path = "zihaojing/mumo-pretrain"
    """

    model_class_str = model_args.model_class
    model_class = globals().get(model_class_str)
    if model_class is None:
        raise ValueError(f"Model class '{model_class_str}' not found in current scope.")

    ### Start Load Model

    if hasattr(model_args, "task_type"):
        config.update({"task_type": model_args.task_type})

    print("Start load model")
    if model_args.model_name_or_path:
        # This supports both local paths and Hugging Face Hub model IDs
        # Hub format: "username/model-name"
        # Local format: "/path/to/model" or "relative/path/to/model"
        print(f"Loading model from: {model_args.model_name_or_path}")

        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            trust_remote_code=True,
            use_flash_attention_2=False,
            token=True if model_args.use_auth_token else None,  # Updated parameter name
        )
    else:
        model = model_class(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        print(
            f"Training new model from scratch - Total size={n_params/2**20:.2f}M params"
        )
        if hasattr(model_args, "task_type") and model_args.task_type is not None:
            raise ValueError(
                "If you are running funetuning tasks you need to sepcify a pretrained model."
            )

    # if tokenizer is not None:
    #     try:
    #         model.resize_token_embeddings(len(tokenizer))
    #     except Exception as err:
    #         print(err)

    print("End load model")
    ### End Load Model
    return model
