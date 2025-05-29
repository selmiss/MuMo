from transformers.models.mamba.modeling_mamba import (
    MambaRMSNorm,
    MambaBlock,
    MambaPreTrainedModel,
)
import torch
from transformers.models.bert.modeling_bert import BertPooler
from typing import Optional
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool
from model.attention_mamba import AttentionMambaBlock
from model.fusion_block import HierarchicalFusionBlock
import torch.nn.init as init
from transformers.modeling_outputs import ModelOutput
from transformers.models.bert.modeling_bert import BertAttention
from model.fusion_block import TransformerFusionBlock

def reverse_seq_tensor(batch_seq):
    # Reverse the sequences along dimension 1 (seq_len)
    reversed_seq = torch.flip(batch_seq, dims=[1])

    return reversed_seq  
    
# MuMo core model - Insert Graph and Geometry Information in the half way.
class MuMoModel(MambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.input_mamba = MambaBlock(config, layer_idx=-1)

        self.layers = nn.ModuleList(
            [
                AttentionMambaBlock(config, layer_idx=idx)
                for idx in range(config.num_hidden_layers // 2)
            ]
        )
        self.graph_layers = nn.ModuleList(
            [
                HierarchicalFusionBlock(config, layer_idx=idx, global_inject=True, brics=config.brics, geo_operation=config.geo_operation)
                for idx in range(config.num_hidden_layers // 2)
            ]
        )
        self.layer_norm = MambaRMSNorm(config.hidden_size)
        self.final_norm = MambaRMSNorm(config.hidden_size)

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, new_embeddings):
        self.embedding = new_embeddings

    def forward(
        self, 
        input_ids, 
        attention_mask: Optional[torch.LongTensor] = None, 
        graph_data: Optional[Data] = None,
        **kwargs
    ):

        embedding = self.embedding(input_ids)
        all_hidden_states = []
        all_attn_score = []
        
        hidden_states = embedding
        
        hidden_states = self.input_mamba(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        hidden_states = self.layer_norm(hidden_states)
        
        for layer in self.layers:
            hidden_states= layer(
                hidden_states=hidden_states, attention_mask=attention_mask
            )
            all_hidden_states.append(hidden_states)
            # all_attn_score.append(attn_score)
        
        for layer in self.graph_layers:
            hidden_states, graph_data= layer(
                hidden_states=hidden_states, attention_mask=attention_mask, graph_data=graph_data
            )
            all_hidden_states.append(hidden_states)
            # all_attn_score.append(attn_score)
           

        hidden_states = self.final_norm(hidden_states)
        
        all_hidden_states = torch.stack(all_hidden_states, dim=1)
        
        return (hidden_states, all_hidden_states, all_attn_score)

# MuMo pretrain model
class MuMoPretrain(MambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layer_hidden_states = []
        self.backbone = MuMoModel(config=config)
        self.to_logits = nn.Sequential(
            nn.Linear(config.hidden_size, config.vocab_size),
        )

    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        graph_data: Optional[Data] = None,
        **kwargs
    ):
        backbone_output = self.backbone(input_ids, attention_mask=attention_mask, graph_data=graph_data)
        logits = self.to_logits(backbone_output[0])

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return {"logits": logits, "loss": loss}
    
class MuMoFormerPretrain(MambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layer_hidden_states = []
        self.backbone = TransformerMuMoModel(config=config)
        self.to_logits = nn.Sequential(
            nn.Linear(config.hidden_size, config.vocab_size),
        )

    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        graph_data: Optional[Data] = None,
        **kwargs
    ):
        backbone_output = self.backbone(input_ids, attention_mask=attention_mask, graph_data=graph_data)
        logits = self.to_logits(backbone_output[0])

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return {"logits": logits, "loss": loss}

# MuMo Fintune Model.
class MuMoFinetune(MambaPreTrainedModel):
    def __init__(self, config, class_weight=None):
        super().__init__(config)

        self.backbone = MuMoModel(config=config)
        self.pooler = BertPooler(config)
        self.pool_method = getattr(config, 'pool_method', 'bert')
        if self.pool_method == "bipooler" or self.pool_method == "mixpooler":
            self.pooler_b = BertPooler(config)
            print("Bipooler or mixpooler Activated.")
        self.class_weight = class_weight
        
        self.task_type = config.task_type
        self.multi_mark = False
        if  self.task_type == "regression":
            self.loss_fn = nn.MSELoss()
            self.output_size = 1
        elif  self.task_type == "classification":
            self.output_size = 2
                
            if self.class_weight is not None:
                self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weight)
            else:
                self.loss_fn = nn.CrossEntropyLoss()
            if config.output_size != -1:
                self.output_size = config.output_size
                self.multi_mark = True
                self.loss_fn = nn.BCEWithLogitsLoss()
        
        self.task_head = nn.Sequential(
            nn.Linear(config.hidden_size, self.output_size),
        )
        if self.pool_method == "bipooler":
            self.task_head = nn.Sequential(
                nn.Linear(2 * config.hidden_size, self.output_size),
            )
        if self.pool_method == "mixpooler":
            self.task_head = nn.Sequential(
                nn.Linear(3 * config.hidden_size, self.output_size),
            )
        # self._initialize_parameters()
        
    def initialize_parameters(self):
        for name, param in self.pooler.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
            elif 'bias' in name:
                init.zeros_(param)
        if self.pool_method == "bipooler" or self.pool_method == "mixpooler":
            for name, param in self.pooler_b.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)
                elif 'bias' in name:
                    init.zeros_(param)
        for layer in self.task_head:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)
                
    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        graph_data: Optional[Data] = None,
        **kwargs
    ):
        backbone_output = self.backbone(input_ids, attention_mask=attention_mask, graph_data=graph_data)
        pooled_output = self.pooler(backbone_output[0])
        if self.pool_method == "bipooler":
            pooled_output_b = self.pooler_b(reverse_seq_tensor(backbone_output[0]))
            pooled_output = torch.cat([pooled_output, pooled_output_b], dim=1)
        elif self.pool_method == "mixpooler":
            pooled_output_b = self.pooler_b(reverse_seq_tensor(backbone_output[0]))
            mean_pool = torch.mean(backbone_output[0], dim=1)
            max_pooled, _ = torch.max(backbone_output[0], dim=1)
            pooled_output = torch.cat([pooled_output, pooled_output_b, mean_pool], dim=1)
            
        logits = self.task_head(pooled_output)

        
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.task_type == "regression":
                logits = logits.view(-1)
                labels = labels.view(-1)
                loss = self.loss_fn(logits, labels)
            elif self.task_type == "classification":
                logits = logits.view(-1, self.output_size)
                if not self.multi_mark:
                    labels = labels.view(-1)
                    labels = labels.long()
                elif self.multi_mark:
                    labels = labels.view(-1, self.output_size)
                    labels = labels.to(dtype=torch.float)

                loss = self.loss_fn(logits, labels)
                

        return {"logits": logits, "loss": loss}

class TransformerMuMoModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Replace input MambaBlock with FFN
        self.input_ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )

        # Replace AttentionMambaBlock with standard transformer blocks
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    BertAttention(config),
                    nn.Sequential(
                        nn.Linear(config.hidden_size, config.intermediate_size),
                        nn.GELU(),
                        nn.Linear(config.intermediate_size, config.hidden_size)
                    )
                )
                for idx in range(config.num_hidden_layers // 2)
            ]
        )

        # Use TransformerFusionBlock instead of HierarchicalFusionBlock
        self.graph_layers = nn.ModuleList(
            [
                TransformerFusionBlock(config, layer_idx=idx, global_inject=True, brics=config.brics, geo_operation=config.geo_operation)
                for idx in range(config.num_hidden_layers // 2)
            ]
        )

        # Replace MambaRMSNorm with LayerNorm
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.final_norm = nn.LayerNorm(config.hidden_size)

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, new_embeddings):
        self.embedding = new_embeddings

    def forward(
        self, 
        input_ids, 
        attention_mask: Optional[torch.LongTensor] = None, 
        graph_data: Optional[Data] = None,
        **kwargs
    ):
        embedding = self.embedding(input_ids)
        all_hidden_states = []
        all_attn_score = []
        
        hidden_states = embedding
        
        # Process through input FFN
        hidden_states = self.input_ffn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        
        # Process through transformer layers
        for layer in self.layers:
            # Self attention
            attention_output = layer[0](
                hidden_states=hidden_states,
                attention_mask=attention_mask[:, None, None, :]
            )
            hidden_states = attention_output[0] + hidden_states
            
            # FFN
            ffn_output = layer[1](hidden_states)
            hidden_states = ffn_output + hidden_states
            
            all_hidden_states.append(hidden_states)
        
        # Process through graph layers
        for layer in self.graph_layers:
            hidden_states, graph_data = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                graph_data=graph_data
            )
            all_hidden_states.append(hidden_states)

        hidden_states = self.final_norm(hidden_states)
        
        all_hidden_states = torch.stack(all_hidden_states, dim=1)
        
        return (hidden_states, all_hidden_states, all_attn_score)