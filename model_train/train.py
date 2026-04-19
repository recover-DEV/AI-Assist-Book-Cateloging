import os
import torch
from utils.read_data import build_dataset, TextDataset
from torch.utils.data import DataLoader
from arch.encoder_decoder import EncoderDecoder
from arch.LSTM import LSTM_encoder, LSTM_decoder
from arch.train_pred import train_seq2seq, predict_seq2seq
from utils.Accurancy import bleu


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join(BASE_DIR, "data_set/book2019-2023.csv")
test_file_path = os.path.join(BASE_DIR, "data_set/book2024.csv")
embedding_size = 256
num_hiddens = 256
num_layers = 2
dropout = 0.2
batch_size = 64
num_steps = 50
lr = 0.0001
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

intro_token_list, classify_token_list = build_dataset(train_file_path)
test_intro_token_list, test_classify_token_list = build_dataset(test_file_path)
min_freq = 1          # 过滤低频词（出现少于1次）
max_freq = None       # 高频词上限由停用词表控制，此处不额外限制；如需数值限制可设为文档数的80%
dataset = TextDataset(intro_token_list, classify_token_list, min_freq=min_freq, max_freq=max_freq, src_num_steps=num_steps, tgt_num_steps=num_steps)
test_dataset = TextDataset(test_intro_token_list, test_classify_token_list, min_freq=min_freq, max_freq=max_freq, src_num_steps=num_steps, tgt_num_steps=num_steps)

src_vocab_size = len(dataset.vocab)
tgt_vocab_size = len(dataset.label_vocab)

data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_data_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

encoder = LSTM_encoder(src_vocab_size, embedding_size, num_hiddens, num_layers, dropout)
decoder = LSTM_decoder(tgt_vocab_size, embedding_size, num_hiddens, num_layers, dropout)

EncoderDecoder = EncoderDecoder(encoder, decoder)
optimizer = torch.optim.Adam(EncoderDecoder.parameters(), lr=lr)

EncoderDecoder.to(device)
save_dir = os.path.join(BASE_DIR, "models")
os.makedirs(save_dir, exist_ok=True)
vocab_path = os.path.join(save_dir, "vocab.pth")
torch.save({
    'src_vocab': dataset.vocab.to_state(),
    'tgt_vocab': dataset.label_vocab.to_state(),
    'num_steps': num_steps,
}, vocab_path)
print(f"已保存词表到 {vocab_path}")
save_epoch = 5
eval_epoch = 5

# 从 models/ 中找训练轮次最多的模型并加载
start_epoch = 0
checkpoints = []
if os.path.isdir(save_dir):
    for fname in os.listdir(save_dir):
        if fname.startswith("model") and fname.endswith(".pth"):
            try:
                epoch_num = int(fname[len("model"):-len(".pth")])
                checkpoints.append((epoch_num, fname))
            except ValueError:
                pass
if checkpoints:
    best_epoch, best_fname = max(checkpoints, key=lambda x: x[0])
    ckpt_path = os.path.join(save_dir, best_fname)
    EncoderDecoder.load_state_dict(torch.load(ckpt_path, map_location=device))
    start_epoch = best_epoch
    print(f"已加载模型 {best_fname}，从第 {start_epoch} 轮继续训练")
else:
    print("未找到已有模型，从第 0 轮开始训练")

train_seq2seq(EncoderDecoder, data_iter, optimizer, num_epochs, dataset.label_vocab, device, save_dir, save_epoch, eval_epoch, start_epoch)


pred_seq,_  = predict_seq2seq(EncoderDecoder, "本书介绍DWDM的发展与起源、系统的网元级测试、系统测试结果举例等。", dataset.vocab, dataset.label_vocab, num_steps, device)
pred_seq = pred_seq.split('<bos>')[-1]
label_seq = 'TN929.11'
print("pred_seq: ", pred_seq)
print("label_seq: ", label_seq)
print(f'BLEU {bleu(pred_seq, label_seq, k=3):.3f}')
