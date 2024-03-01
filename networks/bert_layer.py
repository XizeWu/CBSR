import copy
import math
import sys

import torch
from torch import nn


def gelu(x):  # bert的激活函数
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)  # 求均值
        s = (x - u).pow(2).mean(-1, keepdim=True)  # 求方差
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        # hidden_size必须是num_attention_heads的整数倍，以这里的bert-base为例，
        # 每个attention包含12个head，hidden_size是768，所以每个head大小即attention_head_size=768/12=64
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 12*64=768 之所以不用hidden_size,因为有剪枝prune_heads操作，这里代码没有写

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # 把hidden_size拆成多个头输出的形状，并且将中间两维转置以进行矩阵相乘；
        # x一般就是模型的输入，也就是我们刚才得到的embedding[128, 32, 768]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # [128, 32, 12, 64]
        # 注意这里的写法，x.size()[:-1]的意思是得到x的前两个纬度，shape为[128, 32]
        # print(new_x_shape)就为torch.Size([128, 32, 12, 64])  这个变量就代表一个shape，它不是一个向量
        x = x.view(*new_x_shape)  # [128, 32, 12, 64]  这里是使x变为new_x_shape这个形状
        # *是为什么：多看官方文档  https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        return x.permute(0, 2, 1, 3)        # [128, 12, 32, 64]

    def forward(self, hidden_states, attention_mask):
        # attention_mask：[128,1,1,32]就是bert的输入mask扩充了两个纬度
        # 比如y=[6,3],shape为[2],y.unsqueeze(0)，它的纬度就是[1,2],值为[[6,3]]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # 这里纬度怎么相乘的可以自己算一下
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # (batch_size, num_attention_heads, sequence_length, attention_head_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [128, 12, 32, 32]  (batch_size, num_attention_heads, sequence_length, sequence_length)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 除以根号dk
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask  # 这里的attention_mask是为了对句子进行padding
        # 但输入的mask不是[1, 1, 1, 1, 0, 0]这样的吗，形状[128, 32]，按理应该相乘的
        # 但其实attention_mask被动过手脚了  (BertModel函数的extended_attention_mask处)
        # print一下 [[    -0.,     -0.,     -0.,  ..., -10000., -10000., -10000.]]],..
        #         [[[    -0.,     -0.,     -0.,  ..., -10000., -10000., -10000.]]],..
        # 将原本为1的部分变为0，而原本为0的部分（即padding）变为一个较大的负数，这样相加就得到了一个较大的负值
        # 这样一来经过softmax操作以后这一项就会变成接近0的数，实现了padding的目的

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        # (batch_size, num_attention_heads, sequence_length, attention_head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [128, 32, 768]
        context_layer = context_layer.view(*new_context_layer_shape)  # 多头注意力concat
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 残差连接 对应公式LayerNorm(s+sublayer(x)) 也就是Add&Norm
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):  # bert的两个输入
        # input_tensor.size():[128, 32, 768] attention_mask.size():[128, 1, 1, 32]
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output