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
from model.attention_mamba import AttentionMambaBlock, MambaBlock_without_self_attention
from model.fusion_block import (
    HierarchicalFusionBlock,
    HierarchicalFusionBlock_without_self_attention,
)
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
                HierarchicalFusionBlock(
                    config,
                    layer_idx=idx,
                    global_inject=True,
                    brics=config.brics,
                    geo_operation=config.geo_operation,
                )
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
        **kwargs,
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
            hidden_states = layer(
                hidden_states=hidden_states, attention_mask=attention_mask
            )
            all_hidden_states.append(hidden_states)
            # all_attn_score.append(attn_score)

        if graph_data is not None:
            for layer in self.graph_layers:
                hidden_states, graph_data = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    graph_data=graph_data,
                )
                all_hidden_states.append(hidden_states)
            # all_attn_score.append(attn_score)

        hidden_states = self.final_norm(hidden_states)

        all_hidden_states = torch.stack(all_hidden_states, dim=1)

        return (hidden_states, all_hidden_states, all_attn_score, graph_data)


class MuMoModel_without_self_attention(MambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.input_mamba = MambaBlock(config, layer_idx=-1)

        self.layers = nn.ModuleList(
            [
                MambaBlock_without_self_attention(config, layer_idx=idx)
                for idx in range(config.num_hidden_layers // 2)
            ]
        )
        self.graph_layers = nn.ModuleList(
            [
                HierarchicalFusionBlock_without_self_attention(
                    config,
                    layer_idx=idx,
                    global_inject=True,
                    brics=config.brics,
                    geo_operation=config.geo_operation,
                )
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
        **kwargs,
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
            hidden_states = layer(
                hidden_states=hidden_states, attention_mask=attention_mask
            )
            all_hidden_states.append(hidden_states)
            # all_attn_score.append(attn_score)

        for layer in self.graph_layers:
            hidden_states, graph_data = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                graph_data=graph_data,
            )
            all_hidden_states.append(hidden_states)
            # all_attn_score.append(attn_score)

        hidden_states = self.final_norm(hidden_states)

        all_hidden_states = torch.stack(all_hidden_states, dim=1)

        return (hidden_states, all_hidden_states, all_attn_score, graph_data)


# MuMo pretrain model
class MuMoPretrain(MambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layer_hidden_states = []
        self.no_self_attention = getattr(config, "no_self_attention", False)
        if self.no_self_attention:
            print("Attention-free MuMo Model Used.")
            self.backbone = MuMoModel_without_self_attention(config=config)
        else:
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
        **kwargs,
    ):
        backbone_output = self.backbone(
            input_ids, attention_mask=attention_mask, graph_data=graph_data
        )
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
        **kwargs,
    ):
        backbone_output = self.backbone(
            input_ids, attention_mask=attention_mask, graph_data=graph_data
        )
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

        self.no_self_attention = getattr(config, "no_self_attention", False)
        if self.no_self_attention:
            print("Attention-free MuMo Model Used.")
            self.backbone = MuMoModel_without_self_attention(config=config)
        else:
            self.backbone = MuMoModel(config=config)
        self.pooler = BertPooler(config)
        self.use_graph_embeddings = getattr(config, "use_graph_embeddings", False)
        self.pool_method = getattr(config, "pool_method", "bert")
        if self.pool_method == "bipooler" or self.pool_method == "mixpooler":
            self.pooler_b = BertPooler(config)
            print("Bipooler or mixpooler Activated.")
        self.class_weight = class_weight

        self.task_type = config.task_type
        self.multi_mark = False
        if self.task_type == "regression":
            self.loss_fn = nn.MSELoss()
            self.output_size = 1
            if config.output_size != -1:
                self.output_size = config.output_size
                self.multi_mark = True
        elif self.task_type == "classification":
            self.output_size = 2

            if self.class_weight is not None:
                self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weight)
            else:
                self.loss_fn = nn.CrossEntropyLoss()
            if config.output_size != -1:
                self.output_size = config.output_size
                self.multi_mark = True
                self.loss_fn = nn.BCEWithLogitsLoss()

        hidden_size = config.hidden_size
        if self.use_graph_embeddings:
            hidden_size += config.hidden_size
        self.task_head = nn.Sequential(
            nn.Linear(hidden_size, self.output_size),
        )
        if self.pool_method == "bipooler":
            self.task_head = nn.Sequential(
                nn.Linear(2 * hidden_size, self.output_size),
            )
        if self.pool_method == "mixpooler":
            self.task_head = nn.Sequential(
                nn.Linear(3 * hidden_size, self.output_size),
            )

        # self._initialize_parameters()

    def initialize_parameters(self):
        for name, param in self.pooler.named_parameters():
            if "weight" in name:
                init.xavier_uniform_(param)
            elif "bias" in name:
                init.zeros_(param)
        if self.pool_method == "bipooler" or self.pool_method == "mixpooler":
            for name, param in self.pooler_b.named_parameters():
                if "weight" in name:
                    init.xavier_uniform_(param)
                elif "bias" in name:
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
        **kwargs,
    ):
        backbone_output = self.backbone(
            input_ids, attention_mask=attention_mask, graph_data=graph_data
        )
        pooled_output = self.pooler(backbone_output[0])
        if self.use_graph_embeddings:
            graph_embeddings = global_add_pool(
                backbone_output[3].x, backbone_output[3].batch
            )
            pooled_output = torch.cat([pooled_output, graph_embeddings], dim=1)
        if self.pool_method == "bipooler":
            pooled_output_b = self.pooler_b(reverse_seq_tensor(backbone_output[0]))
            pooled_output = torch.cat([pooled_output, pooled_output_b], dim=1)
        elif self.pool_method == "mixpooler":
            pooled_output_b = self.pooler_b(reverse_seq_tensor(backbone_output[0]))
            mean_pool = torch.mean(backbone_output[0], dim=1)
            max_pooled, _ = torch.max(backbone_output[0], dim=1)
            pooled_output = torch.cat(
                [pooled_output, pooled_output_b, mean_pool], dim=1
            )

        logits = self.task_head(pooled_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.task_type == "regression":
                if not self.multi_mark:  # Single output
                    logits = logits.view(-1)  # [B]
                    labels = labels.view(-1)  # [B]
                else:  # Multi output / Multi task
                    logits = logits.view(-1, self.output_size)  # [B, T]
                    labels = labels.view(-1, self.output_size)  # [B, T]
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


# MuMo Fintune Model.
class MuMoFinetunePairwise(MambaPreTrainedModel):

    def __init__(self, config, class_weight=None):
        super().__init__(config)

        self.backbone = MuMoModel(config=config)
        self.pooler = BertPooler(config)
        self.pool_method = getattr(config, "pool_method", "bert")
        if self.pool_method == "bipooler" or self.pool_method == "mixpooler":
            self.pooler_b = BertPooler(config)
            print("Bipooler or mixpooler Activated.")
        self.class_weight = class_weight
        self.pairwise_margin = config.pairwise_margin
        # Weights for combining pairwise (ranking) and regression (value-based) losses
        self.pairwise_weight = getattr(config, "pairwise_weight", 1.0)
        self.regression_weight = getattr(config, "regression_weight", 1.0)
        self.task_type = config.task_type
        self.multi_mark = False
        if self.task_type == "regression":
            self.loss_fn = nn.MSELoss()
            self.output_size = 1
        elif self.task_type == "classification":
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
            if "weight" in name:
                init.xavier_uniform_(param)
            elif "bias" in name:
                init.zeros_(param)
        if self.pool_method == "bipooler" or self.pool_method == "mixpooler":
            for name, param in self.pooler_b.named_parameters():
                if "weight" in name:
                    init.xavier_uniform_(param)
                elif "bias" in name:
                    init.zeros_(param)
        for layer in self.task_head:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = self(**inputs)
        loss = outputs["loss"]

        if "loss_pairwise" in outputs:
            print(f"loss_pairwise: {outputs['loss_pairwise'].item():.4f}")
        if "loss_regression" in outputs:
            print(f"loss_regression: {outputs['loss_regression'].item():.4f}")

        return (loss, outputs) if return_outputs else loss

    def forward(
        self,
        input_ids_a,
        input_ids_b,
        attention_mask_a: Optional[torch.LongTensor] = None,
        graph_data_a: Optional[Data] = None,
        attention_mask_b: Optional[torch.LongTensor] = None,
        graph_data_b: Optional[Data] = None,
        values_a: Optional[torch.LongTensor] = None,
        values_b: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # Encode first sample
        output_a = self.backbone(
            input_ids_a, attention_mask=attention_mask_a, graph_data=graph_data_a
        )
        pooled_a = self.pooler(output_a[0])
        if self.pool_method == "bipooler":
            pooled_bi_a = self.pooler_b(reverse_seq_tensor(output_a[0]))
            pooled_a = torch.cat([pooled_a, pooled_bi_a], dim=1)
        elif self.pool_method == "mixpooler":
            pooled_bi_a = self.pooler_b(reverse_seq_tensor(output_a[0]))
            mean_pool = torch.mean(output_a[0], dim=1)
            max_pool, _ = torch.max(output_a[0], dim=1)
            pooled_a = torch.cat([pooled_a, pooled_bi_a, mean_pool], dim=1)

        score_a = self.task_head(pooled_a).view(-1)  # regression scalar

        # Encode second sample
        output_b = self.backbone(
            input_ids_b, attention_mask=attention_mask_b, graph_data=graph_data_b
        )
        pooled_b = self.pooler(output_b[0])
        if self.pool_method == "bipooler":
            pooled_bi_b = self.pooler_b(reverse_seq_tensor(output_b[0]))
            pooled_b = torch.cat([pooled_b, pooled_bi_b], dim=1)
        elif self.pool_method == "mixpooler":
            pooled_bi_b = self.pooler_b(reverse_seq_tensor(output_b[0]))
            mean_pool = torch.mean(output_b[0], dim=1)
            max_pool, _ = torch.max(output_b[0], dim=1)
            pooled_b = torch.cat([pooled_b, pooled_bi_b, mean_pool], dim=1)

        score_b = self.task_head(pooled_b).view(-1)

        # Loss: MarginRankingLoss
        pair_loss = None
        total_loss = None
        if labels is not None:
            targets = labels.float() * 2 - 1  # 0/1 → -1/1
            loss_fn = torch.nn.MarginRankingLoss(margin=self.pairwise_margin)
            pair_loss = loss_fn(score_a, score_b, -targets)
            total_loss = pair_loss

        # Regression loss (Mean Absolute Error) between predicted scores and provided values
        reg_loss = None
        if values_a is not None and values_b is not None:
            # Ensure type and device consistency
            if not torch.is_tensor(values_a):
                values_a = torch.tensor(
                    values_a, dtype=score_a.dtype, device=score_a.device
                )
            else:
                values_a = values_a.to(score_a.device, dtype=score_a.dtype)
            if not torch.is_tensor(values_b):
                values_b = torch.tensor(
                    values_b, dtype=score_b.dtype, device=score_b.device
                )
            else:
                values_b = values_b.to(score_b.device, dtype=score_b.dtype)

            mae_a = torch.nn.functional.l1_loss(score_a, values_a, reduction="mean")
            mae_b = torch.nn.functional.l1_loss(score_b, values_b, reduction="mean")
            reg_loss = 0.5 * (mae_a + mae_b)

        # Combine losses if both are available, otherwise use the one that exists
        if pair_loss is not None and reg_loss is not None:
            total_loss = (
                self.pairwise_weight * pair_loss + self.regression_weight * reg_loss
            )
        elif pair_loss is None and reg_loss is not None:
            total_loss = reg_loss
        # If both None, total_loss remains None

        return {
            "logits": torch.stack([score_a, score_b], dim=1),
            "loss": total_loss,
            "loss_pairwise": pair_loss.detach() if pair_loss is not None else None,
            "loss_regression": reg_loss.detach() if reg_loss is not None else None,
        }


class TransformerMuMoModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Replace input MambaBlock with FFN
        # self.input_ffn = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.intermediate_size),
        #     nn.GELU(),
        #     nn.Linear(config.intermediate_size, config.hidden_size)
        # )

        # Replace AttentionMambaBlock with standard transformer blocks
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn_ln": nn.LayerNorm(config.hidden_size),
                        "attn": BertAttention(config),
                        "ffn_ln": nn.LayerNorm(config.hidden_size),
                        "ffn": nn.Sequential(
                            nn.Linear(config.hidden_size, config.intermediate_size),
                            nn.GELU(),
                            nn.Linear(config.intermediate_size, config.hidden_size),
                        ),
                    }
                )
                for _ in range(config.num_hidden_layers // 2)
            ]
        )

        # Use TransformerFusionBlock instead of HierarchicalFusionBlock
        self.graph_layers = nn.ModuleList(
            [
                TransformerFusionBlock(
                    config,
                    layer_idx=idx,
                    global_inject=True,
                    brics=config.brics,
                    geo_operation=config.geo_operation,
                )
                for idx in range(config.num_hidden_layers // 2)
            ]
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
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
        **kwargs,
    ):
        embedding = self.embedding(input_ids)
        all_hidden_states = []
        all_attn_score = []

        hidden_states = embedding

        # Process through transformer layers
        for layer in self.layers:
            if attention_mask is not None and attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]  # [B, 1, 1, L]
            attn_out = layer["attn"](
                hidden_states=layer["attn_ln"](hidden_states),
                attention_mask=attention_mask,
            )
            hidden_states = hidden_states + self.dropout(attn_out[0])
            ffn_out = layer["ffn"](layer["ffn_ln"](hidden_states))
            hidden_states = hidden_states + self.dropout(ffn_out)

            all_hidden_states.append(hidden_states)

        # Process through graph layers
        for layer in self.graph_layers:
            hidden_states, graph_data = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                graph_data=graph_data,
            )
            all_hidden_states.append(hidden_states)

        hidden_states = self.final_norm(hidden_states)

        all_hidden_states = torch.stack(all_hidden_states, dim=1)

        return (hidden_states, all_hidden_states, all_attn_score)


class MuMoFinetuneFormer(MambaPreTrainedModel):
    def __init__(self, config, class_weight=None):
        super().__init__(config)

        self.backbone = TransformerMuMoModel(config=config)
        self.pooler = BertPooler(config)
        self.pool_method = getattr(config, "pool_method", "bert")
        if self.pool_method == "bipooler" or self.pool_method == "mixpooler":
            self.pooler_b = BertPooler(config)
            print("Bipooler or mixpooler Activated.")
        self.class_weight = class_weight

        self.task_type = config.task_type
        self.multi_mark = False
        if self.task_type == "regression":
            self.loss_fn = nn.MSELoss()
            self.output_size = 1
        elif self.task_type == "classification":
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
            if "weight" in name:
                init.xavier_uniform_(param)
            elif "bias" in name:
                init.zeros_(param)
        if self.pool_method == "bipooler" or self.pool_method == "mixpooler":
            for name, param in self.pooler_b.named_parameters():
                if "weight" in name:
                    init.xavier_uniform_(param)
                elif "bias" in name:
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
        **kwargs,
    ):
        backbone_output = self.backbone(
            input_ids, attention_mask=attention_mask, graph_data=graph_data
        )
        pooled_output = self.pooler(backbone_output[0])
        if self.pool_method == "bipooler":
            pooled_output_b = self.pooler_b(reverse_seq_tensor(backbone_output[0]))
            pooled_output = torch.cat([pooled_output, pooled_output_b], dim=1)
        elif self.pool_method == "mixpooler":
            pooled_output_b = self.pooler_b(reverse_seq_tensor(backbone_output[0]))
            mean_pool = torch.mean(backbone_output[0], dim=1)
            max_pooled, _ = torch.max(backbone_output[0], dim=1)
            pooled_output = torch.cat(
                [pooled_output, pooled_output_b, mean_pool], dim=1
            )

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


# MuMo FintuneReaction Model, this was proved to be not useful, don't use it.
class MuMoFinetuneReaction(MambaPreTrainedModel):
    def __init__(self, config, class_weight=None):
        super().__init__(config)

        self.no_self_attention = getattr(config, "no_self_attention", False)
        if self.no_self_attention:
            print("Attention-free MuMo Model Used.")
            self.backbone = MuMoModel_without_self_attention(config=config)
        else:
            self.backbone = MuMoModel(config=config)

        self.num_input_smiles = config.num_input_smiles

        self.pooler = nn.ModuleList()
        for i in range(self.num_input_smiles):
            self.pooler.append(BertPooler(config))

        self.use_graph_embeddings = getattr(config, "use_graph_embeddings", False)
        self.pool_method = getattr(config, "pool_method", "bert")
        if self.pool_method == "bipooler" or self.pool_method == "mixpooler":
            self.pooler_b = BertPooler(config)
            print("Bipooler or mixpooler Activated.")
        self.class_weight = class_weight

        self.task_type = config.task_type
        self.multi_mark = False
        if self.task_type == "regression":
            self.loss_fn = nn.MSELoss()
            self.output_size = 1
            if config.output_size != -1:
                self.output_size = config.output_size
                self.multi_mark = True
        elif self.task_type == "classification":
            self.output_size = 2

            if self.class_weight is not None:
                self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weight)
            else:
                self.loss_fn = nn.CrossEntropyLoss()
            if config.output_size != -1:
                self.output_size = config.output_size
                self.multi_mark = True
                self.loss_fn = nn.BCEWithLogitsLoss()

        hidden_size = config.hidden_size
        if self.use_graph_embeddings:
            hidden_size += config.hidden_size
        self.task_head = nn.Sequential(
            nn.Linear(self.num_input_smiles * hidden_size, self.output_size)
        )
        if self.pool_method == "bipooler":
            self.task_head = nn.Sequential(
                nn.Linear(2 * hidden_size, self.output_size),
            )
        if self.pool_method == "mixpooler":
            self.task_head = nn.Sequential(
                nn.Linear(3 * hidden_size, self.output_size),
            )

        # self._initialize_parameters()

    def initialize_parameters(self):
        for name, param in self.pooler.named_parameters():
            if "weight" in name:
                init.xavier_uniform_(param)
            elif "bias" in name:
                init.zeros_(param)
        if self.pool_method == "bipooler" or self.pool_method == "mixpooler":
            for name, param in self.pooler_b.named_parameters():
                if "weight" in name:
                    init.xavier_uniform_(param)
                elif "bias" in name:
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
        **kwargs,
    ):
        pooled_outputs = []
        for i in range(self.num_input_smiles):
            import pdb

            pdb.set_trace()
            backbone_output = self.backbone(
                input_ids[i], attention_mask=attention_mask[i], graph_data=graph_data[i]
            )
            pooled_output = self.pooler[i](backbone_output[0])
            pooled_outputs.append(pooled_output)

        pooled_output = torch.cat(pooled_outputs, dim=1)

        logits = self.task_head(pooled_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.task_type == "regression":
                if not self.multi_mark:  # Single output
                    logits = logits.view(-1)  # [B]
                    labels = labels.view(-1)  # [B]
                else:  # Multi output / Multi task
                    logits = logits.view(-1, self.output_size)  # [B, T]
                    labels = labels.view(-1, self.output_size)  # [B, T]
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
