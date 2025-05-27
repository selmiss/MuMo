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

    Args:
        config (_type_): _description_
        tokenizer (_type_, optional): _description_. Defaults to None.
        training_args (_type_, optional): _description_. Defaults to None.
        model_args (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        model
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

        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            trust_remote_code=True,
            use_flash_attention_2=False,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = model_class(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        print(
            f"Training new model from scratch - Total size={n_params/2**20:.2f}M params"
        )
        if hasattr(model_args, "task_type") and model_args.task_type is not None:
            raise ValueError("If you are running funetuning tasks you need to sepcify a pretrained model.")

    # if tokenizer is not None:
    #     try:
    #         model.resize_token_embeddings(len(tokenizer))
    #     except Exception as err:
    #         print(err)
        
    print( "End load model")
    ### End Load Model
    return model

def load_model_from_path(model_path, model_class: str):
    """
    Load model from it's dir path.

    Args:
        model_path (_type_): _description_
        model_class (str): _description_

    Returns:
        model
    """
    
    model_class_str = model_class
    model_class = globals().get(model_class_str)
    if model_class is None:
        raise ValueError(f"Model class '{model_class_str}' not found in current scope.")
    
    config = AutoConfig.from_pretrained(model_path)
    model = model_class.from_pretrained(
            model_path,
            from_tf=bool(".ckpt" in model_path),
            config=config,
            use_flash_attention_2=False,
        )
    return model
    