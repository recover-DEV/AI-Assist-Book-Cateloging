import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.read_data import build_bert_dataset
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from arch.BERT import BERT_encoder
from arch.BERT_classifier import (
    BERTHierarchicalClassifier,
    HierarchicalLabelDataset,
    build_all_level_vocabs,
    predict_hierarchical,
)
from utils.Accurancy import bleu, hierarchical_accuracy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join(BASE_DIR, "data_set/book2019-2023.csv")

from utils.read_data import BERTTextDataset
from arch.encoder_decoder import EncoderDecoder
from arch.Transformer import TransformerDecoder
from arch.train_pred import train_bert_seq2seq, predict_bert_seq2seq

# ── 数据 ──────────────────────────────────────────────────────────────────────
batch_size    = 64
src_num_steps = 128
tgt_num_steps = 15
val_ratio     = 0.05

# ── 训练 ──────────────────────────────────────────────────────────────────────
lr_bert    = 1e-5
lr_decoder = 5e-5
num_epochs = 20
save_epoch = 1
eval_epoch = 1

# ── 模型结构 ──────────────────────────────────────────────────────────────────
n_freeze_layers = 6
n_heads         = 8
decoder_layers  = 4
dim_feedforward = 1024
dropout         = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

title_list, intro_list, classify_list = build_bert_dataset(train_file_path)
encoder = BERT_encoder(model_path='E:\\dev\\项目\\AI_Assist_book_cateloging\\model_train\\bert-base-chinese', max_length=src_num_steps, n_freeze_layers=n_freeze_layers)
bert_dataset = BERTTextDataset(encoder, title_list, intro_list, classify_list,
                               min_freq=0, tgt_num_steps=tgt_num_steps)

train_data, val_data = random_split(
    bert_dataset,
    [1 - val_ratio, val_ratio],
    generator=torch.Generator().manual_seed(42),
)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)

decoder = TransformerDecoder(
    len(bert_dataset.label_vocab),
    d_model=768, n_heads=n_heads, decoder_layers=decoder_layers,
    dim_feedforward=dim_feedforward, dropout=dropout,
)
model = EncoderDecoder(encoder, decoder)

bert_params  = [p for n, p in model.named_parameters() if p.requires_grad and "encoder.bert" in n]
other_params = [p for n, p in model.named_parameters() if p.requires_grad and "encoder.bert" not in n]
optimizer = torch.optim.Adam([
    {"params": bert_params,  "lr": lr_bert},
    {"params": other_params, "lr": lr_decoder},
])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

save_dir = os.path.join(BASE_DIR, "models")
os.makedirs(save_dir, exist_ok=True)
bert_vocab_path = os.path.join(save_dir, "bert_vocab.pth")
torch.save({
    'tgt_vocab': bert_dataset.label_vocab.to_state(),
    'src_num_steps': src_num_steps, 'tgt_num_steps': tgt_num_steps,
    'n_freeze_layers': n_freeze_layers, 'n_heads': n_heads,
    'decoder_layers': decoder_layers, 'dim_feedforward': dim_feedforward,
    'dropout': dropout,
}, bert_vocab_path)
print(f"已保存词表到 {bert_vocab_path}")

start_epoch = 0
checkpoints = []
if os.path.isdir(save_dir):
    for fname in os.listdir(save_dir):
        if fname.startswith("model") and fname.endswith(".pth") and fname != "model_best.pth":
            try:
                checkpoints.append((int(fname[len("model"):-len(".pth")]), fname))
            except ValueError:
                pass
if checkpoints:
    best_epoch, best_fname = max(checkpoints, key=lambda x: x[0])
    model.load_state_dict(torch.load(os.path.join(save_dir, best_fname), map_location='cpu'))
    model.to(device)
    start_epoch = best_epoch
    print(f"已加载模型 {best_fname}，从第 {start_epoch} 轮继续训练")
else:
    print("未找到已有模型，从第 0 轮开始训练")

train_bert_seq2seq(
    model, train_loader, optimizer, num_epochs,
    bert_dataset.label_vocab, device, save_dir, save_epoch, eval_epoch,
    start_epoch=start_epoch, val_iter=val_loader, scheduler=scheduler,
)
pred_seq, _ = predict_bert_seq2seq(
    model, "DWDM系统测试",
    "本书介绍DWDM的发展与起源、系统的网元级测试、系统测试结果举例等。",
    bert_dataset.label_vocab, tgt_num_steps, device=device,
)
pred_seq  = pred_seq.split('<bos>')[-1]
label_seq = 'TN929.11'
print("pred_seq: ", pred_seq)
print("label_seq:", label_seq)
print(f'BLEU {bleu(pred_seq, label_seq, k=3):.3f}')



# # 层级判别式模型（BERT + 多层级分类头）

# # ── 超参数 ────────────────────────────────────────────────────────────────────
# batch_size      = 64
# src_num_steps   = 128        
# val_ratio       = 0.05      

# lr_bert         = 2e-5      
# lr_head         = 1e-4       
# num_epochs      = 20
# save_epoch      = 1
# eval_epoch      = 1

# n_freeze_layers = 4          
# dropout         = 0.1


# LEVELS          = [1, 2, 3, 4, 5, None]
# # 各层级损失权重（越细粒度权重越高，full code 权重最大）
# level_weights   = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]

# device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# save_dir = os.path.join(BASE_DIR, "models_cls")
# os.makedirs(save_dir, exist_ok=True)

# # ── 数据加载 ──────────────────────────────────────────────────────────────────
# print("读取训练数据 …")
# title_list, intro_list, classify_list = build_bert_dataset(train_file_path)
# print(f"  共 {len(title_list)} 条样本")

# encoder = BERT_encoder(model_path='E:\\dev\\项目\\AI_Assist_book_cateloging\\model_train\\bert-base-chinese', max_length=src_num_steps, n_freeze_layers=n_freeze_layers)

# print("构建层级词表 …")
# level_vocabs      = build_all_level_vocabs(classify_list, LEVELS)
# level_vocab_sizes = [len(v) for v in level_vocabs]
# for lv, v in zip(LEVELS, level_vocabs):
#     tag = str(lv) if lv is not None else "full"
#     print(f"  level={tag:>4s}: {len(v):>5d} 类")

# dataset = HierarchicalLabelDataset(
#     encoder, title_list, intro_list, classify_list, level_vocabs, LEVELS
# )

# train_data, val_data = random_split(
#     dataset,
#     [1 - val_ratio, val_ratio],
#     generator=torch.Generator().manual_seed(42),
# )
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)

# # ── 模型构建 ──────────────────────────────────────────────────────────────────
# model = BERTHierarchicalClassifier(encoder, level_vocab_sizes, dropout=dropout)

# bert_params = [p for n, p in model.named_parameters() if p.requires_grad and "encoder.bert" in n]
# head_params = [p for n, p in model.named_parameters() if p.requires_grad and "encoder.bert" not in n]
# optimizer = torch.optim.Adam([
#     {"params": bert_params, "lr": lr_bert},
#     {"params": head_params, "lr": lr_head},
# ])
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

# # ── 词表 + 配置持久化（eval 时用于重建模型结构）────────────────────────────────
# vocab_path = os.path.join(save_dir, "hierarchical_vocab.pth")
# torch.save({
#     "level_vocabs":      level_vocabs,
#     "levels":            LEVELS,
#     "level_vocab_sizes": level_vocab_sizes,
#     "src_num_steps":     src_num_steps,
#     "n_freeze_layers":   n_freeze_layers,
#     "dropout":           dropout,
# }, vocab_path)
# print(f"已保存层级词表到 {vocab_path}")

# # ── 断点续训 ──────────────────────────────────────────────────────────────────
# start_epoch = 0
# checkpoints = []
# if os.path.isdir(save_dir):
#     for fname in os.listdir(save_dir):
#         if fname.startswith("model") and fname.endswith(".pth") and fname != "model_best.pth":
#             try:
#                 checkpoints.append((int(fname[len("model"):-len(".pth")]), fname))
#             except ValueError:
#                 pass
# if checkpoints:
#     best_epoch, best_fname = max(checkpoints, key=lambda x: x[0])
#     model.load_state_dict(torch.load(os.path.join(save_dir, best_fname), map_location='cpu'))
#     model.to(device)
#     start_epoch = best_epoch
#     print(f"已加载 {best_fname}，从第 {start_epoch} 轮继续")
# else:
#     print("未找到已有模型，从第 0 轮开始训练")



# def _eval_accuracy(net, val_iter, dev):
#     net.eval()
#     correct = total = 0
#     with torch.no_grad():
#         for batch in val_iter:
#             ids, attn, tti, *labels = [x.to(dev) for x in batch]
#             logits_list = net(ids, attn, tti)
#             pred = logits_list[-1].argmax(dim=-1)   # full code 头
#             correct += (pred == labels[-1]).sum().item()
#             total   += ids.size(0)
#     net.train()
#     return correct / total if total > 0 else 0.0


# model.to(device)
# best_val_acc   = 0.0
# best_ckpt_path = None

# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss    = 0.0
#     epoch_correct = 0
#     n_samples     = 0

#     for batch in train_loader:
#         ids, attn, tti, *label_list = [x.to(device) for x in batch]
#         logits_list = model(ids, attn, tti)

#         # 加权多任务损失：各层级交叉熵之和
#         loss = sum(
#             w * F.cross_entropy(logits, labels)
#             for w, logits, labels in zip(level_weights, logits_list, label_list)
#         )

#         optimizer.zero_grad()
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()

#         epoch_loss    += loss.item() * ids.size(0)
#         epoch_correct += (logits_list[-1].argmax(dim=-1) == label_list[-1]).sum().item()
#         n_samples     += ids.size(0)

#     if (epoch + 1) % save_epoch == 0:
#         ckpt = os.path.join(save_dir, f"model{epoch + start_epoch + 1}.pth")
#         torch.save(model.state_dict(), ckpt)
#         print(f"model saved to {ckpt}")

#     if (epoch + 1) % eval_epoch == 0:
#         train_acc  = epoch_correct / max(n_samples, 1)
#         train_loss = epoch_loss    / max(n_samples, 1)
#         val_acc    = _eval_accuracy(model, val_loader, device)

#         # scheduler 以 val loss 为准（1 - acc 作为代理损失）
#         scheduler.step(1 - val_acc)

#         log = (
#             f"epoch {epoch + 1}, "
#             f"train_loss {train_loss:.4f}, train_acc {train_acc:.4f}, "
#             f"val_acc {val_acc:.4f}, "
#             f'lr {optimizer.param_groups[0]["lr"]:.2e}'
#         )
#         if val_acc > best_val_acc:
#             best_val_acc   = val_acc
#             best_ckpt_path = os.path.join(save_dir, "model_best.pth")
#             torch.save(model.state_dict(), best_ckpt_path)
#             log += "  best"
#         print(log)

# if best_ckpt_path:
#     print(f"最佳模型 {best_ckpt_path}，val_acc={best_val_acc:.4f}")

# pred  = predict_hierarchical(
#     model,
#     "DWDM系统测试",
#     "本书介绍DWDM的发展与起源、系统的网元级测试、系统测试结果举例等。",
#     level_vocabs,
#     device,
# )
# label = "TN929.11"
# print(f"pred : {pred}")
# print(f"label: {label}")
# print(f"BLEU : {bleu(pred, label, k=3):.3f}")
