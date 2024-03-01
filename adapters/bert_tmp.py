import logging

import torch
import torch.nn as nn
from transformers import BertModel
from transformers.models.bert.modeling_bert import ACT2FN, BertSelfOutput

from adapters.common import AdapterConfig


logging.basicConfig(level=logging.INFO)


class BertAdapter(nn.Module):
    def __init__(self, config: AdapterConfig):
        super(BertAdapter, self).__init__()
        # self.activation = nn.ReLU()
        self.dpout = nn.Dropout(0.5)

        self.down_project = nn.Linear(config.hidden_size, config.adapter_size)
        nn.init.normal_(self.down_project.weight, std=config.adapter_initializer_range)
        nn.init.zeros_(self.down_project.bias)

        self.down_lynorm = nn.LayerNorm(config.adapter_size, eps=1e-12)
        # self.down_lynorm.bias.data.zero_()
        # self.down_lynorm.weight.data.fill_(1.0)

        self.up_project = nn.Linear(config.adapter_size, config.hidden_size)
        nn.init.normal_(self.up_project.weight, std=config.adapter_initializer_range)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, hidden_states: torch.Tensor):
        down_projected = self.down_project(hidden_states)

        down_projected = self.down_lynorm(down_projected)

        activated = self.dpout(down_projected)

        up_projected = self.up_project(activated)

        return hidden_states + up_projected


class BertAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 self_output: BertSelfOutput,
                 config: AdapterConfig):
        super(BertAdaptedSelfOutput, self).__init__()
        self.self_output = self_output
        self.adapter = BertAdapter(config)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        adapter_out = self.adapter(input_tensor)
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states) + adapter_out
        hidden_states = self.self_output.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def adapt_bert_self_output(config: AdapterConfig):
    return lambda self_output: BertAdaptedSelfOutput(self_output, config=config)


def add_bert_adapters(bert_model: BertModel, config: AdapterConfig) -> BertModel:
    for layer in bert_model.encoder.layer:
        layer.attention.output = adapt_bert_self_output(config)(layer.attention.output)
        layer.output = adapt_bert_self_output(config)(layer.output)
    return bert_model

    # for layer in bert_model.encoder.layer:
    #     layer.attention.output = adapt_bert_self_output(config)(layer.attention.output)
    #     layer.output = adapt_bert_self_output(config)(layer.output)
    # return bert_model


def unfreeze_bert_adapters(bert_model: nn.Module) -> nn.Module:
    # Unfreeze trainable parts — layer norms and adapters
    for name, sub_module in bert_model.named_modules():
        if isinstance(sub_module, (BertAdapter, nn.LayerNorm)):
            for param_name, param in sub_module.named_parameters():
                param.requires_grad = True
    return bert_model
