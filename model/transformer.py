import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, hidden_size, dropout, num_layers=1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Implementation of improved part
        self.lstm = nn.LSTM(d_model, hidden_size, num_layers=num_layers, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size * 2, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: # (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        # src: [B, C, T]
        # src_w = src.permute(0, 2, 1)
        src_w = src
        src2 = self.self_attn(src_w, src_w, src_w, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src_w = src_w + self.dropout1(src2)
        src_w = self.norm1(src_w)
        src2 = self.linear(self.dropout(self.activation(self.lstm(src_w)[0])))
        src_w = src_w + self.dropout2(src2)
        src_w = self.norm2(src_w)
        return src_w # output: [B, C, T]


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
