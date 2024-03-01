import torch.nn as nn


class Encoder_am(nn.Module):
    def __init__(self, dim_in, dim_hidden, input_dp=0.2, rnn_dp=0, n_layers=1, bidirectional=True, rnn_cell='gru'):
        super(Encoder_am, self).__init__()
        self.dim_video = dim_in
        self.input_dropout_p = input_dp
        self.rnn_dropout_p = rnn_dp
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        if bidirectional:
            self.dim_hidden = dim_hidden // 2

        self.vid2hid = nn.Sequential(nn.Linear(self.dim_video, self.dim_hidden),
                                     nn.GELU(),
                                     nn.LayerNorm(self.dim_hidden, eps=1e-12),
                                     )

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn = self.rnn_cell(self.dim_hidden, self.dim_hidden, self.n_layers, batch_first=True,
                                 bidirectional=self.bidirectional, dropout=self.rnn_dropout_p)
        self.v_input_ln = nn.LayerNorm((self.dim_hidden*2 if bidirectional else self.self.dim_hidden), eps=1e-12,
                                       elementwise_affine=False)
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_normal_(self.vid2hid[0].weight)

    def forward(self, vid_feats):
        batch_size, num_frame, dim_video = vid_feats.size()

        vid_feats_trans = self.vid2hid(vid_feats.view(-1, self.dim_video))
        vid_feats = vid_feats_trans.view(batch_size, num_frame, -1)

        self.rnn.flatten_parameters()
        foutput, fhidden = self.rnn(vid_feats)

        foutput = self.v_input_ln(foutput)

        return foutput