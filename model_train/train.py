import os
import torch
from utils.read_data import build_dataset, TextDataset
from torch.utils.data import DataLoader
from arch.encoder_decoder import EncoderDecoder
from arch.LSTM import train_seq2seq, predict_seq2seq, LSTM_encoder, LSTM_decoder
from utils.Accurancy import bleu


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join(BASE_DIR, "data_set/book2019-2023.csv")
test_file_path = os.path.join(BASE_DIR, "data_set/book2024.csv")
embedding_size = 256
num_hiddens = 256
num_layers = 2
dropout = 0.1
batch_size = 64
num_steps = 30
lr = 0.005
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

intro_token_list, classify_token_list = build_dataset(train_file_path)
test_intro_token_list, test_classify_token_list = build_dataset(test_file_path)
dataset = TextDataset(intro_token_list, classify_token_list, min_freq=0, src_num_steps=num_steps, tgt_num_steps=num_steps)
test_dataset = TextDataset(test_intro_token_list, test_classify_token_list, min_freq=0, src_num_steps=num_steps, tgt_num_steps=num_steps)

src_vocab_size = len(dataset.vocab)
tgt_vocab_size = len(dataset.label_vocab)

data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_data_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

encoder = LSTM_encoder(src_vocab_size, embedding_size, num_hiddens, num_layers, dropout)
decoder = LSTM_decoder(tgt_vocab_size, embedding_size, num_hiddens, num_layers, dropout)
optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

EncoderDecoder = EncoderDecoder(encoder, decoder)
EncoderDecoder.to(device)

# train_seq2seq(EncoderDecoder, data_iter, lr, num_epochs, dataset.label_vocab, device)
pred_seq,_  = predict_seq2seq(EncoderDecoder, "这本书是关于用户研究与用户体验设计的，用户研究是用户体验设计中非常重要的一部分。", dataset.vocab, dataset.label_vocab, num_steps, device)
label_seq = pred_seq
print("pred_seq: ", pred_seq)
print("label_seq: ", label_seq)
print(f'BLEU {bleu(pred_seq, label_seq, k=2):.3f}')
