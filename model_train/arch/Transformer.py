from torch import nn
from arch.encoder_decoder import Encoder, Decoder
import torch

class TransformerEncoder(Encoder):
    """
    Transformer Encoder
    Args:
        src_vocab_size: 源词表大小
        d_model: 特征维度
        n_heads: 多头注意力头数
        encoder_layers: 编码器层数
        dim_feedforward: 前馈神经网络维度
        dropout: dropout率
    """
    def __init__(self,src_vocab_size, d_model, n_heads, encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.max_len = src_vocab_size
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, encoder_layers)

    def forward(self, src, valid_len, *args):
        src = self.embedding(src).permute(1,0,2)

        # src形状为（seq_len, batch_size, d_model）
        # valid_len形状为（batch_size,）
        # print(src.shape)
        mask = torch.arange(src.shape[0]).unsqueeze(0) >= valid_len.unsqueeze(1)

        # mask形状为（batch_size, seq_len）
        # mask为True的位置会被忽略
        src = self.transformer_encoder(src, src_key_padding_mask=mask)
        return src

class TransformerDecoder(Decoder):
    """
    Transformer Decoder
    Args:
        tgt_vocab_size: 目标词表大小
        d_model: 特征维度
        n_heads: 多头注意力头数
        decoder_layers: 解码器层数
        dim_feedforward: 前馈神经网络维度
        dropout: dropout率
    """
    def __init__(self, tgt_vocab_size, d_model, n_heads, decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.transformer_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_layer, decoder_layers)

    def init_state(self, memory, *args):
        return memory

    def forward(self, tgt, memory):
        # tgt形状为（batch_size, seq_len）
        # memory形状为（seq_len, batch_size, d_model）
        tgt = self.embedding(tgt).permute(1,0,2)

        print(tgt.shape)
        # tgt形状为（seq_len, batch_size, d_model）
        tgt = self.transformer_decoder(tgt, memory)
        return tgt