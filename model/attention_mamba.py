from torch import nn
from torch_geometric.nn import global_mean_pool
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.mamba.modeling_mamba import (
    MambaRMSNorm,
    MambaBlock,
)
import torch
from typing import Optional


class MambaBlock_without_self_attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        # MambaBlock
        self.ssm_block = MambaBlock(config, layer_idx=layer_idx)
        self.layer_norm = MambaRMSNorm(config.hidden_size)
        # self.attention = BertAttention(config)

    def forward(self, hidden_states, attention_mask: Optional[torch.LongTensor] = None):

        residual = hidden_states

        # attention_output = self.attention(hidden_states=hidden_states, attention_mask=attention_mask[:, None, None, :])

        # hidden_states = attention_output[0] + residual

        hidden_states = self.ssm_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class AttentionMambaBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        # MambaBlock
        self.ssm_block = MambaBlock(config, layer_idx=layer_idx)
        self.layer_norm = MambaRMSNorm(config.hidden_size)
        self.attention = BertAttention(config)

    def forward(self, hidden_states, attention_mask: Optional[torch.LongTensor] = None):

        residual = hidden_states

        attention_output = self.attention(
            hidden_states=hidden_states, attention_mask=attention_mask[:, None, None, :]
        )

        hidden_states = attention_output[0] + residual

        hidden_states = self.ssm_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states
