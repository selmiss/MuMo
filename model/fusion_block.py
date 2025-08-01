from model.module import graph2batch_sequence, sequence2batch_graph
from torch import nn
from model.geo_layer import GeoGraphLyaerMPNN
from torch_geometric.nn import global_add_pool
import torch
from typing import Optional
from torch_geometric.data import Data
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.mamba.modeling_mamba import (
    MambaRMSNorm,
    MambaBlock,
)


class HierarchicalFusionBlock_without_self_attention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx,
        global_inject=False,
        num_graph_layers=2,
        graph_enabled=True,
        brics=True,
        geo_operation=True,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.graph_hidden_size = config.hidden_size
        self.global_inject = global_inject
        self.num_graph_layers = num_graph_layers
        self.graph_enabled = graph_enabled
        # MambaBlock
        self.ssm_block = MambaBlock(config, layer_idx=layer_idx)

        self.first_graph_layer = GeoGraphLyaerMPNN(
            9, 3, config.hidden_size, brics=brics, geo_operation=geo_operation
        )
        self.graph_layers = torch.nn.ModuleList(
            [
                GeoGraphLyaerMPNN(
                    config.hidden_size,
                    config.hidden_size,
                    config.hidden_size,
                    brics=brics,
                    geo_operation=geo_operation,
                )
                for i in range(self.num_graph_layers)
            ]
        )

        # Hidden size projection.
        # self.resize_graph_hidden = nn.Linear(config.graph_hidden_size, config.hidden_size)
        # self.back_graph_hidden = nn.Linear(config.hidden_size, config.graph_hidden_size)

        # Norm layers
        self.layer_norm = MambaRMSNorm(config.hidden_size)

        # Attentions
        # self.attention = BertAttention(config)
        self.graph_cross_attention = BertAttention(config)
        self.sequence_cross_attention = BertAttention(config)

        # Global Ijection
        if self.global_inject:
            self.graph_map_layer = nn.Linear(
                config.hidden_size + self.graph_hidden_size, config.hidden_size
            )

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.LongTensor] = None,
        graph_data: Optional[Data] = None,
    ):

        # Main stream pass the self-attention
        residual = hidden_states
        # attention_output = self.attention(hidden_states=hidden_states, attention_mask=attention_mask[:, None, None, :])
        # hidden_states = attention_output[0] + residual

        # Process if this is the first graph layer.
        if graph_data.x.shape[1] != self.graph_hidden_size:
            graph_data = self.first_graph_layer(graph_data)

        # Use bidirectional cross attention.
        if self.graph_enabled:
            graph_hidden_states, graph_attention_mask = graph2batch_sequence(
                graph_hidden_states=graph_data.x, batch=graph_data.batch
            )
            # graph_hidden_states = self.resize_graph_hidden(graph_hidden_states)

            hidden_states_backup = hidden_states.clone()
            hidden_states = self.sequence_cross_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask[:, None, None, :],
                encoder_hidden_states=graph_hidden_states,
                encoder_attention_mask=graph_attention_mask[:, None, None, :],
            )[0]
            graph_hidden_states = self.graph_cross_attention(
                hidden_states=graph_hidden_states,
                attention_mask=graph_attention_mask[:, None, None, :],
                encoder_hidden_states=hidden_states_backup,
                encoder_attention_mask=attention_mask[:, None, None, :],
            )[0]

            # graph_hidden_states = self.back_graph_hidden(graph_hidden_states)
            graph_data.x = sequence2batch_graph(
                graph_hidden_states, graph_attention_mask
            )

            # Process the graph hidden states
            for layer in self.graph_layers:
                graph_data = layer(graph_data)

        # After cross attention, the main stream states will pass the mamba block
        hidden_states = self.ssm_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )

        # Concat and inplace operation.
        if self.global_inject and self.graph_enabled:
            graph_embeddings = global_add_pool(graph_data.x, graph_data.batch)
            cls_token = hidden_states[:, 0, :]
            concat_embeddings = torch.cat((graph_embeddings, cls_token), dim=-1)
            updated_embedding = self.graph_map_layer(concat_embeddings)
            hidden_states_clone = hidden_states.clone()
            hidden_states_clone[:, 0, :] = cls_token + updated_embedding
            hidden_states = hidden_states_clone + residual

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states, graph_data


class HierarchicalFusionBlock(nn.Module):
    def __init__(
        self,
        config,
        layer_idx,
        global_inject=False,
        num_graph_layers=2,
        graph_enabled=True,
        brics=True,
        geo_operation=True,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.graph_hidden_size = config.hidden_size
        self.global_inject = global_inject
        self.num_graph_layers = num_graph_layers
        self.graph_enabled = graph_enabled
        # MambaBlock
        self.ssm_block = MambaBlock(config, layer_idx=layer_idx)

        self.first_graph_layer = GeoGraphLyaerMPNN(
            9, 3, config.hidden_size, brics=brics, geo_operation=geo_operation
        )
        self.graph_layers = torch.nn.ModuleList(
            [
                GeoGraphLyaerMPNN(
                    config.hidden_size,
                    config.hidden_size,
                    config.hidden_size,
                    brics=brics,
                    geo_operation=geo_operation,
                )
                for i in range(self.num_graph_layers)
            ]
        )

        # Hidden size projection.
        # self.resize_graph_hidden = nn.Linear(config.graph_hidden_size, config.hidden_size)
        # self.back_graph_hidden = nn.Linear(config.hidden_size, config.graph_hidden_size)

        # Norm layers
        self.layer_norm = MambaRMSNorm(config.hidden_size)

        # Attentions
        self.attention = BertAttention(config)
        self.graph_cross_attention = BertAttention(config)
        self.sequence_cross_attention = BertAttention(config)

        # Global Ijection
        if self.global_inject:
            self.graph_map_layer = nn.Linear(
                config.hidden_size + self.graph_hidden_size, config.hidden_size
            )

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.LongTensor] = None,
        graph_data: Optional[Data] = None,
    ):

        # Main stream pass the self-attention
        residual = hidden_states
        attention_output = self.attention(
            hidden_states=hidden_states, attention_mask=attention_mask[:, None, None, :]
        )
        hidden_states = attention_output[0] + residual

        # Process if this is the first graph layer.
        if graph_data.x.shape[1] != self.graph_hidden_size:
            graph_data = self.first_graph_layer(graph_data)

        # Use bidirectional cross attention.
        if self.graph_enabled:
            graph_hidden_states, graph_attention_mask = graph2batch_sequence(
                graph_hidden_states=graph_data.x, batch=graph_data.batch
            )
            # graph_hidden_states = self.resize_graph_hidden(graph_hidden_states)

            hidden_states_backup = hidden_states.clone()
            hidden_states = self.sequence_cross_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask[:, None, None, :],
                encoder_hidden_states=graph_hidden_states,
                encoder_attention_mask=graph_attention_mask[:, None, None, :],
            )[0]
            graph_hidden_states = self.graph_cross_attention(
                hidden_states=graph_hidden_states,
                attention_mask=graph_attention_mask[:, None, None, :],
                encoder_hidden_states=hidden_states_backup,
                encoder_attention_mask=attention_mask[:, None, None, :],
            )[0]

            # graph_hidden_states = self.back_graph_hidden(graph_hidden_states)
            graph_data.x = sequence2batch_graph(
                graph_hidden_states, graph_attention_mask
            )

            # Process the graph hidden states
            for layer in self.graph_layers:
                graph_data = layer(graph_data)

        # After cross attention, the main stream states will pass the mamba block
        hidden_states = self.ssm_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )

        # Concat and inplace operation.
        if self.global_inject and self.graph_enabled:
            graph_embeddings = global_add_pool(graph_data.x, graph_data.batch)
            cls_token = hidden_states[:, 0, :]
            concat_embeddings = torch.cat((graph_embeddings, cls_token), dim=-1)
            updated_embedding = self.graph_map_layer(concat_embeddings)
            hidden_states_clone = hidden_states.clone()
            hidden_states_clone[:, 0, :] = cls_token + updated_embedding
            hidden_states = hidden_states_clone + residual

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states, graph_data


class TransformerFusionBlock(nn.Module):
    def __init__(
        self,
        config,
        layer_idx,
        global_inject=False,
        num_graph_layers=2,
        graph_enabled=True,
        brics=True,
        geo_operation=True,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.graph_hidden_size = config.hidden_size
        self.global_inject = global_inject
        self.num_graph_layers = num_graph_layers
        self.graph_enabled = graph_enabled

        # FFN instead of MambaBlock
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )

        self.first_graph_layer = GeoGraphLyaerMPNN(
            9, 3, config.hidden_size, brics=brics, geo_operation=geo_operation
        )
        self.graph_layers = torch.nn.ModuleList(
            [
                GeoGraphLyaerMPNN(
                    config.hidden_size,
                    config.hidden_size,
                    config.hidden_size,
                    brics=brics,
                    geo_operation=geo_operation,
                )
                for i in range(self.num_graph_layers)
            ]
        )

        # Norm layers
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.attn_ln = nn.LayerNorm(config.hidden_size)
        self.ffn_ln = nn.LayerNorm(config.hidden_size)

        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        # Attentions
        self.attention = BertAttention(config)
        self.graph_cross_attention = BertAttention(config)
        self.sequence_cross_attention = BertAttention(config)

        # Global Injection
        if self.global_inject:
            self.graph_map_layer = nn.Linear(
                config.hidden_size + self.graph_hidden_size, config.hidden_size
            )

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.LongTensor] = None,
        graph_data: Optional[Data] = None,
    ):
        # Main stream pass the self-attention
        residual = hidden_states
        attention_output = self.attention(
            hidden_states=self.attn_ln(hidden_states), attention_mask=attention_mask
        )
        hidden_states = attention_output[0] + residual

        # Process if this is the first graph layer.
        if graph_data.x.shape[1] != self.graph_hidden_size:
            graph_data = self.first_graph_layer(graph_data)

        # Use bidirectional cross attention.
        if self.graph_enabled:
            graph_hidden_states, graph_attention_mask = graph2batch_sequence(
                graph_hidden_states=graph_data.x, batch=graph_data.batch
            )

            hidden_states_backup = hidden_states.clone()
            hidden_states = self.sequence_cross_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=graph_hidden_states,
                encoder_attention_mask=graph_attention_mask[:, None, None, :],
            )[0]
            graph_hidden_states = self.graph_cross_attention(
                hidden_states=graph_hidden_states,
                attention_mask=graph_attention_mask[:, None, None, :],
                encoder_hidden_states=hidden_states_backup,
                encoder_attention_mask=attention_mask,
            )[0]

            graph_data.x = sequence2batch_graph(
                graph_hidden_states, graph_attention_mask
            )

            # Process the graph hidden states
            for layer in self.graph_layers:
                graph_data = layer(graph_data)

        # FFN instead of MambaBlock
        # ffn_output = self.ffn(self.ffn_ln(hidden_states))
        # hidden_states = ffn_output + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_ln(hidden_states)
        hidden_states = self.dropout(self.ffn(hidden_states)) + residual

        # Concat and inplace operation.
        if self.global_inject and self.graph_enabled:
            graph_embeddings = global_add_pool(graph_data.x, graph_data.batch)
            cls_token = hidden_states[:, 0, :]
            concat_embeddings = torch.cat((graph_embeddings, cls_token), dim=-1)
            updated_embedding = self.graph_map_layer(concat_embeddings)
            hidden_states_clone = hidden_states.clone()
            hidden_states_clone[:, 0, :] = cls_token + updated_embedding
            hidden_states = hidden_states_clone + residual

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states, graph_data
