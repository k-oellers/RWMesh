import argparse
import math
import torch
import torch.nn as nn
from einops import repeat
import models
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Optional, Tuple, Union
from torch import Tensor
import util


class TransformerEncoderLayerVis(TransformerEncoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation="relu",
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, device=None, dtype=None) -> None:
        super(TransformerEncoderLayerVis, self).__init__(d_model, nhead, dim_feedforward, dropout, activation,
                                                         layer_norm_eps, batch_first, device, dtype)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights


class TransformerEncoderVis(TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None, visualize=False):
        super(TransformerEncoderVis, self).__init__(encoder_layer, num_layers, norm)
        self.visualize = visualize

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None):
        """
        Use Transformer Encoder and return attention weight per layer

        :param src: input tensor
        :param mask: optional mask tensor
        :param src_key_padding_mask: optional padding tensor
        :return: output and attention weights
        """
        attn_weights_mat = None
        if self.visualize:
            attn_weights_mat = torch.zeros((src.shape[1], len(self.layers), src.shape[0], src.shape[0]),
                                           device=src.device)
        for i, mod in enumerate(self.layers):
            src, att_weights = mod(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            if self.visualize:
                attn_weights_mat[:, i, :, :] = att_weights

        if self.norm is not None:
            src = self.norm(src)
        return src, attn_weights_mat if self.visualize else None


class TransformerModel(nn.Module):
    """
    Similar to the RNN encoder of the MeshWalker approach but with a Transformer encoder instead of RNNs.
    This allows a parallel processing of the data and visualization of the attention weights.
    """
    def __init__(self, opt: argparse.Namespace, feature_dim: int = 3, classify: bool = True):
        """
        :param opt: Input arguments
        :param feature_dim: size of the feature dimension. Defaults to 3 for the xyz-Position of the vertices. Can be
        increased by using a spectral descriptor.
        :param classify: if True, use classify mode, if False, use segmentation mode. Defaults to True.
        """
        super(TransformerModel, self).__init__()

        self.opt = opt
        self.pos_mlp = models.Projection(layers=[feature_dim, opt.model_size * 2, opt.model_size])
        self.pos_encoder = PositionalEncoding(opt.model_size)
        # use VisTransformer which can return the attention weights
        encoder_layers = TransformerEncoderLayerVis(opt.model_size, opt.nheads, opt.hidden_dim, dropout=0)
        self.transformer_encoder = TransformerEncoderVis(encoder_layers, opt.nlayers, visualize=opt.visualize)
        self.fc = nn.Linear(opt.model_size, opt.n_classes).to(opt.device)
        self.cls_token = nn.Parameter(torch.randn(1, 1, opt.model_size)) if classify else None
        self.classify = classify
        self.use_fc = not (opt.self_supervised and (not opt.pretrained and opt.phase != 'test'))

    def forward(self, src: torch.Tensor, pad: Optional[torch.Tensor] = None) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Call the transformer model

        :param src: List of vertices with shape (batch_size, walks, vertices, feature_dim) or
        (batch_size, vertices, feature_dim)
        :param pad: Optional padding with shape (batch_size, walks, vertices) or (batch_size, vertices)
        :return: When classifying: a tensor of shape (batch_size, walks, classes)
                 When segmenting: a tensor of shape (batch_size, walks, vertices, classes)
                 When self-supervised and classifying: a tensor of shape (batch_size, walks, model_size)
                 When self-supervised and segmenting: a tensor of shape (batch_size, walks, vertices, model_size)
        """
        input_shape = src.shape
        # (batch_size, walks, vertices, dim) -> (batch_size * walks, vertices, dim)
        if len(input_shape) > 3:
            src = util.union_walks(src)
            if pad is not None:
                pad = util.union_walks(pad)

        src = src.permute(1, 0, 2)
        src = self.pos_mlp(src)

        if self.classify:
            src, pad = self.add_cls_token(src, pad, src.shape[1])
        if self.opt.pos_encoding:
            # add position encoding
            src = self.pos_encoder(src)
        output, attn_mat = self.transformer_encoder(src, src_key_padding_mask=pad)

        if self.classify:
            # get CLS token
            output = output[0, :, :]
        else:
            output = output.permute(1, 0, 2)

        if self.use_fc:
            # use fc head
            output = self.fc(output)

        # (batch_size * walks, vertices, dim) -> (batch_size, walks, vertices, dim)
        if len(input_shape) > 3:
            output = util.segregate_walks(output, input_shape[1])
            if attn_mat is not None:
                attn_mat = util.segregate_walks(attn_mat, input_shape[1])

        return (output, attn_mat) if self.opt.visualize else output

    def add_cls_token(self, src: torch.Tensor, pad: torch.Tensor, b: int):
        # add CLS token
        cls_tokens = repeat(self.cls_token, 'n () d -> n b d', b=b)
        src = torch.cat((cls_tokens, src), dim=0)
        # expand padding for CLS token
        pad = torch.cat((torch.zeros((pad.shape[0], 1)).to(pad.device), pad), dim=1)
        return src, pad


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the elements in the transformer encoder
    """

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 1024) -> None:
        """
        Initialize the position arrays

        :param d_model: model size
        :param dropout: dropout rate
        :param max_len: maximum sequence length
        """

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input

        :param src: input tensor
        :return: input tensor with added positional encoding
        """

        output = src + repeat(self.pe[:src.size(0), :], "s () e -> s b e", b=src.size(1))
        return self.dropout(output)