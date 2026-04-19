from arch.encoder_decoder import Encoder, Decoder, EncoderDecoder
from utils.text_handle import intro_tokenize_text, WordCount,classify_tokenize_text
from torch.nn import LSTM
import torch
import os
from torch import nn

class LSTM_encoder(Encoder):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers,dropout=0.0):
        super(LSTM_encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
    
    def forward(self,X, *args):
        X = self.embedding(X).permute(1,0,2)
        output, state = self.rnn(X)
        return output, state

class LSTM_decoder(Decoder):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers,dropout=0.0):
        super(LSTM_decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = LSTM(embedding_size+hidden_size, hidden_size, num_layers, dropout=dropout)
        self.dense = nn.Linear(hidden_size, vocab_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def init_state(self,enc_outputs, *args):
        return enc_outputs[1]
    
    def forward(self,X,state):
        X = self.embedding(X).permute(1,0,2)
        context = state[0][-1].repeat(X.shape[0],1,1)
        X = torch.cat((X,context), 2)
        output, state = self.rnn(X,state)
        output = self.layer_norm(output)
        output = self.dense(output).permute(1,0,2)
        return output, state