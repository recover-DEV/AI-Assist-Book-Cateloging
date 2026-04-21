import os
import torch
from utils.read_data import build_bert_dataset, BERTTextDataset
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from arch.encoder_decoder import EncoderDecoder
from arch.Transformer import TransformerDecoder
from arch.train_pred import train_bert_seq2seq, predict_bert_seq2seq
from utils.Accurancy import bleu
from arch.BERT import BERT_encoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join(BASE_DIR, "data_set/book2019-2023.csv")
batch_size = 64
src_num_steps = 128
tgt_num_steps = 15
lr = 0.0003
num_epochs = 20
val_ratio = 0.05      # 5% 作为验证集
save_epoch = 1
eval_epoch = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

title_list, intro_list, classify_list = build_bert_dataset(train_file_path)
encoder = BERT_encoder(max_length=src_num_steps,freeze_bert=True)
bert_dataset = BERTTextDataset(encoder, title_list, intro_list, classify_list, min_freq=0, tgt_num_steps=tgt_num_steps)

train_data, val_data = random_split(bert_dataset, [1 - val_ratio, val_ratio])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

n_heads = 8
num_layers = 2
dim_feedforward = 512
dropout = 0.3

decoder = TransformerDecoder(len(bert_dataset.label_vocab), d_model=768, n_heads=n_heads, decoder_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout)
model = EncoderDecoder(encoder, decoder)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
save_dir = os.path.join(BASE_DIR, "models")
os.makedirs(save_dir, exist_ok=True)
bert_vocab_path = os.path.join(save_dir, "bert_vocab.pth")
torch.save({
    'tgt_vocab': bert_dataset.label_vocab.to_state(),
    'src_num_steps': src_num_steps,
    'tgt_num_steps': tgt_num_steps,
}, bert_vocab_path)
print(f"已保存词表到 {bert_vocab_path}")


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
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    start_epoch = best_epoch
    print(f"已加载模型 {best_fname}，从第 {start_epoch} 轮继续训练")
else:
    print("未找到已有模型，从第 0 轮开始训练")

train_bert_seq2seq(model, train_loader, optimizer, num_epochs, bert_dataset.label_vocab, device, save_dir, save_epoch, eval_epoch, start_epoch=start_epoch, val_iter=val_loader, scheduler=scheduler)

pred_seq,_  = predict_bert_seq2seq(model, "DWDM系统测试", "本书介绍DWDM的发展与起源、系统的网元级测试、系统测试结果举例等。", bert_dataset.label_vocab, tgt_num_steps, device=device)
pred_seq = pred_seq.split('<bos>')[-1]
label_seq = 'TN929.11'
print("pred_seq: ", pred_seq)
print("label_seq: ", label_seq)
print(f'BLEU {bleu(pred_seq, label_seq, k=3):.3f}')