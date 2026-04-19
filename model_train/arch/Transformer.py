from torch import nn
from arch.encoder_decoder import Encoder, Decoder
import torch
import math

def generate_tgt_mask(T, device):
    """生成布尔型的下三角掩码，形状 (T, T)"""
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    return mask

def generate_square_subsequent_mask(sz):
    """生成下三角掩码，形状 (sz, sz)"""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class PositionalEncoding(nn.Module):
    """Positional encoding.

    Defined in :numref:`sec_self-attention-and-positional-encoding`"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

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

        self.d_model = d_model
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward, dropout, batch_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, encoder_layers,enable_nested_tensor=False)

    def forward(self, src, valid_len, *args):
        src = self.embedding(src)

        # src形状为（batch_size, seq_len, d_model）
        # valid_len形状为（batch_size,）
        src = self.positional_encoding(src * math.sqrt(self.d_model))
        mask = torch.arange(src.shape[1], device=src.device).unsqueeze(0) >= valid_len.unsqueeze(1)

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
        self.d_model = d_model
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        transformer_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(transformer_layer, decoder_layers)
        
        self.dense = nn.Linear(d_model, tgt_vocab_size)

    def init_state(self, memory, enc_valid_len, *args):
        enc_valid_len = torch.arange(memory.shape[1], device=memory.device).unsqueeze(0) >= enc_valid_len.unsqueeze(1)
        return [memory, enc_valid_len]

    def forward(self, tgt, state, tgt_key_padding_mask=None):
        # tgt形状为（batch_size, seq_len）
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt * math.sqrt(self.d_model))

        memory, enc_valid_len = state

        # tgt_mask形状为（seq_len, seq_len）生成下三角掩码，用于掩盖未来的token
        tgt_mask = generate_tgt_mask(tgt.shape[1], device=tgt.device)

        # tgt形状为（batch_size, seq_len, d_model）
        # memory形状为（batch_size, seq_len, d_model）
        tgt = self.transformer_decoder(
            tgt, memory=memory, tgt_mask=tgt_mask, memory_key_padding_mask=enc_valid_len, tgt_key_padding_mask=tgt_key_padding_mask
        )

        # tgt形状为（batch_size, seq_len, d_model）
        tgt = self.dense(tgt)

        # tgt形状为（batch_size, seq_len, tgt_vocab_size）
        return tgt,state