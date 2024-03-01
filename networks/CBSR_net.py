import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.EncoderVid import EncoderVid
from networks.Encoder_appmot import Encoder_am
import math
from torch.nn.parameter import Parameter
from networks.bert_layer import BertAttention
from adapters.common import AdapterConfig, freeze_all_parameters
from adapters.bert_tmp import add_bert_adapters, unfreeze_bert_adapters
from transformers import BertModel


def get_mask(lengths, max_length):
    """ Computes a batch of padding masks given batched lengths """
    mask = 1 * (torch.arange(max_length).unsqueeze(1).to(lengths.device) < lengths).transpose(0, 1)
    return mask


def Simple_CMAtten(x, y):
    # x:[bs, t1, dim]  y:[bs, t2, dim]
    assert 3 == len(x.shape) and 3 == len(y.shape), "[wxz] Shape of x and y are not 3!"
    s = torch.bmm(x, y.transpose(1, 2))
    s_weight = F.softmax(s, dim=-1)
    u_tile = torch.bmm(s_weight, y)

    return u_tile, s_weight


class MultiheadAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.dropout = nn.Dropout(p=dropout)

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.k_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.v_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.out_lin = nn.Linear(in_features=self.dim, out_features=self.dim)

    def forward(self, query, key, value, key_padding_mask, attn_mask=None, output_attentions=False):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        key_padding_mask: torch.tensor(bs, seq_length)
        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        scores_ = scores.mean(1)
        if key_padding_mask is not None:
            key_padding_mask = (
                (~key_padding_mask).view(mask_reshp).expand_as(scores))  # (bs, n_heads, q_length, k_length)
            scores = scores.masked_fill(key_padding_mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if attn_mask is not None:
            weights = weights * attn_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return context, weights.mean(1)
        else:
            return context


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=768, nheads=12, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nheads, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nheads, dropout=dropout)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, feat1, feat2):
        feat1_tmp = self.self_attn(query=feat1, key=feat1, value=feat1, attn_mask=None, key_padding_mask=None)
        feat1 = feat1 + self.dropout1(feat1_tmp)
        feat1 = self.norm1(feat1)

        feat1_tmp = self.multihead_attn(query=feat1, key=feat2, value=feat2, attn_mask=None, key_padding_mask=None)
        feat1 = feat1 + self.dropout2(feat1_tmp)
        feat1 = self.norm2(feat1)

        feat1_tmp = self.linear2(self.dropout(F.relu(self.linear1(feat1))))
        feat1 = feat1 + self.dropout3(feat1_tmp)
        feat1 = self.norm3(feat1)

        return feat1


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, skip=True):
        super(GraphConvolution, self).__init__()
        self.skip = skip
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.bmm(input, self.weight.unsqueeze(0).expand(input.shape[0], -1, -1))
        output = torch.bmm(adj, support)
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand(input.shape[0], -1, -1)
        if self.skip:
            output += support

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class my_GNN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, dropout):
        super(my_GNN, self).__init__()
        self.skip = True

        self.fc_x0 = nn.Linear(dim_in, dim_hidden)
        self.fc_x1 = nn.Linear(dim_in, dim_hidden)
        self.fc_x2 = nn.Linear(dim_in, dim_hidden)

        self.fc_x0q = nn.Linear(dim_in, dim_hidden)
        self.fc_x0k = nn.Linear(dim_in, dim_hidden)

        dim_hidden = dim_out if num_layers == 1 else dim_hidden
        self.gcs_1to0 = nn.ModuleList([GraphConvolution(dim_in, dim_hidden)])
        self.gcs_2to0 = nn.ModuleList([GraphConvolution(dim_in, dim_hidden)])
        self.gcs_0to0 = nn.ModuleList([GraphConvolution(dim_in, dim_hidden)])

        for i in range(num_layers - 1):
            dim_tmp = dim_out if i == num_layers - 2 else dim_hidden
            self.gcs_1to0.append(GraphConvolution(dim_hidden, dim_tmp))
            self.gcs_2to0.append(GraphConvolution(dim_hidden, dim_tmp))
            self.gcs_0to0.append(GraphConvolution(dim_hidden, dim_tmp))

        self.dropout = dropout

        self.gate_1and2 = nn.Sequential(
            nn.Linear(dim_in * 3, 1),
            nn.Sigmoid(),
        )

        self.lynorm = nn.LayerNorm(dim_in, eps=1e-12)

    def construct_cross_graph(self, x0, x1, x2):
        tmp_x0 = self.fc_x0(x0)
        tmp_x1 = self.fc_x1(x1)
        tmp_x2 = self.fc_x2(x2)

        s_1to0 = torch.bmm(tmp_x0, tmp_x1.transpose(1, 2))
        s_2to0 = torch.bmm(tmp_x0, tmp_x2.transpose(1, 2))

        a1to0_weight = F.softmax(s_1to0, dim=2)  # [B, t1, t2]
        a2to0_weight = F.softmax(s_2to0, dim=2)  # [B, t1, t2]

        return a1to0_weight, a2to0_weight

    def construct_self_graph(self, x0):
        tmp_x0q = self.fc_x0q(x0)
        tmp_x0k = self.fc_x0k(x0)

        s_self = torch.bmm(tmp_x0q, tmp_x0k.transpose(1, 2))
        self_weight = F.softmax(s_self, dim=2)         # [B, t1, t2]

        return self_weight

    def forward(self, x0, x1, x2, c_sent):
        output_x0 = x0

        for gc1to0, gc2to0 in zip(self.gcs_1to0, self.gcs_2to0):
            adj_sim_1to0, adj_sim_2to0 = self.construct_cross_graph(output_x0, x1, x2)
            tmp_x1tox0 = F.relu(gc1to0(x1, adj_sim_1to0))
            tmp_x1tox0 = F.dropout(tmp_x1tox0, self.dropout, training=self.training)
            tmp_x2tox0 = F.relu(gc2to0(x2, adj_sim_2to0))
            tmp_x2tox0 = F.dropout(tmp_x2tox0, self.dropout, training=self.training)

            gate_to0 = self.gate_1and2(torch.cat((tmp_x1tox0, c_sent, tmp_x2tox0), dim=-1))
            gcn_out = gate_to0 * tmp_x1tox0 + (1.0 - gate_to0) * tmp_x2tox0
            gcn_out = x0 + gcn_out
            output_x0 = F.dropout(gcn_out, self.dropout, training=self.training)

        adj_sim_self = self.construct_self_graph(output_x0)
        for gc0to0 in self.gcs_0to0:
            tmp_self = F.relu(gc0to0(output_x0, adj_sim_self))
            output_x0 = F.dropout(tmp_self, self.dropout, training=self.training)

        return output_x0


class CACR(nn.Module):
    def __init__(self, dim_in, dpout, video_dim, pos_hidden, alpha, config_bert):
        super(CACR, self).__init__()
        self.dim_in = dim_in
        self.dropout = dpout
        self.alpha = alpha

        self.encode_vid = EncoderVid(feat_dim=video_dim, bbox_dim=5, feat_hidden=self.dim_in, pos_hidden=pos_hidden)

        self.encoder_f = Encoder_am(video_dim, self.dim_in, input_dp=self.dropout, rnn_dp=0,
                                    n_layers=1, bidirectional=True, rnn_cell='gru')
        self.encoder_m = Encoder_am(video_dim, self.dim_in, input_dp=self.dropout, rnn_dp=0,
                                    n_layers=1, bidirectional=True, rnn_cell='gru')

        self.gnn_obj = my_GNN(dim_in=self.dim_in, dim_hidden=self.dim_in, dim_out=self.dim_in, num_layers=2, dropout=0.15)
        self.gnn_vf = my_GNN(dim_in=self.dim_in, dim_hidden=self.dim_in, dim_out=self.dim_in, num_layers=2, dropout=0.15)
        self.gnn_vm = my_GNN(dim_in=self.dim_in, dim_hidden=self.dim_in, dim_out=self.dim_in, num_layers=2, dropout=0.15)

        self.con_obj2app = nn.Sequential(
            nn.Linear(self.dim_in * 2, self.dim_in),
            nn.ELU(inplace=True),
        )
        self.con_obj2mot = nn.Sequential(
            nn.Linear(self.dim_in * 2, self.dim_in),
            nn.ELU(inplace=True),
        )
        self.dout = nn.Dropout(0.5)

    def Sim_obj_by_lan(self, lan, len_lan, vo, len_vo):
        """ Input:
                lan: shape[bs, 1, dim],
                vo:  shape[bs, num_cf, num_r, dim]
            Output:
                Sim  shape[bs * num_cf, num_r]
         """
        bs, num_f, num_r, hdim = vo.shape
        vo = vo.view(bs, num_f * num_r, hdim)
        s = torch.bmm(lan, vo.transpose(1, 2))  # [bs, 1, 768]*[bs, 768, numf*r] -> [bs, 1, num_cf*num_r]
        s_mask = s.data.new(*s.size()).fill_(1).bool()
        for i, (l_1, l_2) in enumerate(zip(len_lan, len_vo)):
            s_mask[i][:l_1, :l_2] = 0  # 设为False保护起来，Ture的被mask掉
        s.data.masked_fill_(s_mask.data, -float("inf"))
        s = s.squeeze().view(bs * num_f, num_r)
        a_weight = F.softmax(s, dim=-1)
        a_weight.data.masked_fill_(a_weight.data != a_weight.data, 0)

        return a_weight

    def Sim_vid_by_lan(self, lan, len_lan, video, len_v):
        s = torch.bmm(lan, video.transpose(1, 2))
        s_mask = s.data.new(*s.size()).fill_(1).bool()  # [B, T1, T2]
        for i, (l_1, l_2) in enumerate(zip(len_lan, len_v)):
            s_mask[i][:l_1, :l_2] = 0
        s.data.masked_fill_(s_mask.data, -float("inf"))
        a_weight = F.softmax(s.squeeze(), dim=-1)
        a_weight.data.masked_fill_(a_weight.data != a_weight.data, 0)

        return a_weight

    def decoupling_obj2(self, video_i, language, alpha):
        '''
            inpout:
                video_i:        shape[bs, num_f, num_r, dim]
                language:   shape[bs, 1, dim]
            output:
                obj:        shape[bs, num_f, num_r*topN, dim]
        '''
        bs_lsent, seq_l, _ = language.shape
        bs_video, num_f, num_r, dim_v = video_i.shape
        assert bs_lsent == bs_video, "[wxz] bs_qsent != bs_video when decoupling!"
        lsent_len = torch.tensor([seq_l] * bs_lsent, dtype=torch.long).cuda()
        video_len = torch.tensor([num_f * num_r] * bs_video, dtype=torch.long).cuda()

        Sim_video_lan = self.Sim_obj_by_lan(language, lsent_len, video_i, video_len)  # [bs*num_cf, num_r]

        _, nvl = Sim_video_lan.sort(descending=True)
        vid_pos_num = int(alpha * num_r)
        idx_max_vid = nvl[:, :vid_pos_num].sort()[0].contiguous().view(-1)  # del all contiguous() for test!
        # idx_max_vid = nvl[:, :vid_pos_num].contiguous().view(-1)  # del all contiguous() for test!
        video_i = video_i.view(bs_video * num_f, num_r, dim_v)
        vid_pos = video_i[torch.arange(bs_video * num_f).view(-1, 1).repeat(1, vid_pos_num).view(-1), idx_max_vid, :]
        vid_pos = vid_pos.view(bs_video * num_f, vid_pos_num, dim_v)

        return vid_pos

    def decoupling(self, video_i, language, topN):
        bs_lsent, seq_l, _ = language.shape         # [bs, 1, dim]
        bs_video, seq_v, dim_v = video_i.shape      # [bs, 10, dim]
        lsent_len = torch.tensor([seq_l] * bs_lsent, dtype=torch.long).cuda()
        video_len = torch.tensor([seq_v] * bs_video, dtype=torch.long).cuda()

        Sim_video_lan = self.Sim_vid_by_lan(language, lsent_len, video_i, video_len)  # [bs_video, seq_v]

        _, nvl = Sim_video_lan.sort(descending=True)
        vid_pos_num = int(topN * seq_v)
        idx_max_vid = nvl[:, :vid_pos_num].sort()[0].contiguous().view(-1)
        vid_pos = video_i[torch.arange(bs_video).view(-1, 1).repeat(1, vid_pos_num).view(-1), idx_max_vid, :]
        vid_pos = vid_pos.view(bs_video, vid_pos_num, dim_v)

        return vid_pos

    def forward(self, video_o, video_f, video_m, qus_feat, know_feat, con_feat, k_sent, q_sent, c_sent):
        video_o = self.encode_vid(video_o)          # [bs, num_cf, num_r, dim]
        video_f = self.encoder_f(video_f)           # [bs, num_v, dim]
        video_m = self.encoder_m(video_m)           # [bs, num_v, dim]
        bsize_o, num_f, num_r, hdim_o = video_o.shape
        bsize_v, num_v, hdim_v = video_f.shape

        ''' obj-level '''
        video_o = video_o.view(bsize_o, num_f * num_r, hdim_o)
        video_o_k = Simple_CMAtten(video_o, know_feat)[0] + video_o
        video_o_q = Simple_CMAtten(video_o, qus_feat)[0] + video_o
        video_o_c = Simple_CMAtten(video_o, con_feat)[0] + video_o

        video_o_k = video_o_k.view(bsize_o, num_f, num_r, hdim_o)
        video_o_q = video_o_q.view(bsize_o, num_f, num_r, hdim_o)
        video_o_c = video_o_c.view(bsize_o, num_f, num_r, hdim_o)
        obj_kpos = self.decoupling_obj2(video_o_k, k_sent, self.alpha)  # [bsize_o*num_cf, num_pos/num_neg, hdim_o]
        obj_qpos = self.decoupling_obj2(video_o_q, q_sent, self.alpha)  # [bsize_o*num_cf, num_pos/num_neg, hdim_o]
        obj_cpos = self.decoupling_obj2(video_o_c, c_sent, self.alpha)  # [bsize_o*num_cf, num_pos/num_neg, hdim_o]

        gate_c1 = c_sent.expand(-1, num_f, -1).reshape(bsize_o * num_f, -1).unsqueeze(dim=1).expand(-1, obj_kpos.shape[1], -1)

        gcn_obj1_pos = self.gnn_obj(obj_cpos, obj_qpos, obj_kpos, gate_c1) + obj_cpos

        obj1_pos = gcn_obj1_pos.mean(dim=1)
        obj1_pos = obj1_pos.view(bsize_o, num_f, hdim_o)  # [bs, num_cf, dim]

        video_f = self.con_obj2app(torch.cat((video_f, obj1_pos), dim=-1))
        video_m = self.con_obj2mot(torch.cat((video_m, obj1_pos), dim=-1))

        ''' video-level '''
        video_f_k = Simple_CMAtten(video_f, know_feat)[0] + video_f
        video_f_q = Simple_CMAtten(video_f, qus_feat)[0] + video_f
        video_f_c = Simple_CMAtten(video_f, con_feat)[0] + video_f

        vf_kpos = self.decoupling(video_f_k, k_sent, self.alpha)
        vf_qpos = self.decoupling(video_f_q, q_sent, self.alpha)
        vf_cpos = self.decoupling(video_f_c, c_sent, self.alpha)

        gate_c2 = c_sent.expand(-1, vf_cpos.shape[1], -1)

        vf_pos = self.gnn_vf(vf_cpos, vf_qpos, vf_kpos, gate_c2) + vf_cpos

        video_m_k = Simple_CMAtten(video_m, know_feat)[0] + video_m
        video_m_q = Simple_CMAtten(video_m, qus_feat)[0] + video_m
        video_m_c = Simple_CMAtten(video_m, con_feat)[0] + video_m

        vm_kpos = self.decoupling(video_m_k, k_sent, self.alpha)
        vm_qpos = self.decoupling(video_m_q, q_sent, self.alpha)
        vm_cpos = self.decoupling(video_m_c, c_sent, self.alpha)

        vm_pos = self.gnn_vm(vm_cpos, vm_qpos, vm_kpos, gate_c2) + vm_cpos

        return vf_pos, vm_pos


class ASCD(nn.Module):
    def __init__(self, dim_in, config_bert):
        super(ASCD, self).__init__()
        self.config_bert = config_bert

        self.gate_appmot = nn.Sequential(
            nn.Linear(dim_in * 3, 1),
            nn.Sigmoid(),
        )

        self.trans_qk = BertAttention(config_bert)
        self.trans_1 = TransformerDecoderLayer()

    def forward(self, q_sent, vf_pos, vm_pos, con_feat):
        gate_qus = q_sent.expand(-1, vm_pos.shape[1], -1)                           # [bs, len_v, dim]
        gate_am = self.gate_appmot(torch.cat((vf_pos, gate_qus, vm_pos), dim=-1))   # [bs, len_v, 1]

        v_all = gate_am * vf_pos + (1.0 - gate_am) * vm_pos

        qk_len = torch.tensor([con_feat.shape[1]] * con_feat.shape[0], dtype=torch.long).cuda()
        mask_qk = get_mask(qk_len, con_feat.shape[1]).cuda()
        extended_attention_mask_qk = mask_qk.unsqueeze(1).unsqueeze(2)
        extended_attention_mask_qk = extended_attention_mask_qk.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask_qk = (1.0 - extended_attention_mask_qk) * -10000.0
        qk_feat = self.trans_qk(con_feat, extended_attention_mask_qk)

        feat_out = self.trans_1(v_all, qk_feat)

        return feat_out


def bert_function(bert, input_tokens):
    bs, nans, lans = input_tokens.shape       # [64,5,29]
    input_tokens = input_tokens.view(bs * nans, lans)   # [320,29]
    attention_mask = (input_tokens > 0).float()

    input_embeds = bert.embeddings(input_tokens)
    output_text = bert(inputs_embeds=input_embeds, attention_mask=attention_mask)
    text_word = output_text[0]

    text_sent = text_word[:, 0, :]
    text_sent = text_sent.view(bs, nans, -1)  # [64,5,768]

    return text_sent


class MyModel(nn.Module):
    def __init__(self, dim_in, dpout, video_dim, pos_hidden, alpha, bert_config):
        super(MyModel, self).__init__()

        bert = BertModel.from_pretrained("bert-base-uncased", config=bert_config)
        config_adapters = AdapterConfig(hidden_size=768, adapter_size=256, adapter_initializer_range=1e-3)
        self.bert_candi = add_bert_adapters(bert, config_adapters)
        self.bert_candi = freeze_all_parameters(self.bert_candi)
        self.bert_candi = unfreeze_bert_adapters(self.bert_candi)

        self.CACR = CACR(dim_in, dpout, video_dim, pos_hidden, alpha, bert_config)
        self.ASCD = ASCD(dim_in, bert_config)

    def forward(self, video_o, video_f, video_m, candi_token, qus_feat, know_feat, con_feat):
        candi_sent = bert_function(self.bert_candi, candi_token)

        q_sent = qus_feat.mean(dim=1).unsqueeze(dim=1)      # [bs, 1, dim]
        k_sent = know_feat.sum(dim=1).unsqueeze(dim=1)      # [bs, 1, dim]
        c_sent = con_feat.sum(dim=1).unsqueeze(dim=1)       # [bs, 1, dim]

        feat_app, feat_mot = self.CACR(video_o, video_f, video_m, qus_feat, know_feat, con_feat, k_sent, q_sent, c_sent)
        feat_out = self.ASCD(c_sent, feat_app, feat_mot, con_feat)

        return feat_out.mean(dim=1), candi_sent