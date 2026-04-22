import os
import torch
import pandas as pd

from arch.BERT import BERT_encoder
from arch.BERT_classifier import BERTHierarchicalClassifier, predict_hierarchical
from utils.Accurancy import bleu, hierarchical_accuracy


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
test_file_path = os.path.join(BASE_DIR, "data_set/book2024.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 层级判别式模型路径 ──────────────────────────────────────────────────────────
models_dir = os.path.join(BASE_DIR, "models_cls")
vocab_path = os.path.join(models_dir, "hierarchical_vocab.pth")

if not os.path.exists(vocab_path):
    raise FileNotFoundError(
        f"未找到词表：{vocab_path}\n请先运行 train_BRET.py 训练并保存 hierarchical_vocab.pth"
    )

cfg = torch.load(vocab_path, map_location="cpu")
level_vocabs = cfg["level_vocabs"]
levels = cfg["levels"]
level_vocab_sizes = cfg["level_vocab_sizes"]
src_num_steps = cfg.get("src_num_steps",128)
n_freeze_layers = cfg.get("n_freeze_layers",4)
dropout = cfg.get("dropout",0.1)

print(f"full code 类别数：{level_vocab_sizes[-1]}")


encoder = BERT_encoder(
    model_path='E:\\dev\\项目\\AI_Assist_book_cateloging\\model_train\\bert-base-chinese',
    max_length=src_num_steps,
    n_freeze_layers=n_freeze_layers,
)
model = BERTHierarchicalClassifier(encoder, level_vocab_sizes, dropout=dropout)


best_path = os.path.join(models_dir, "model_best.pth")
if not os.path.isfile(best_path):
    checkpoints = []
    for fname in os.listdir(models_dir):
        if fname.startswith("model") and fname.endswith(".pth") and fname != "model_best.pth":
            try:
                checkpoints.append((int(fname[len("model"):-len(".pth")]), fname))
            except ValueError:
                pass
    if not checkpoints:
        raise FileNotFoundError(f"在 {models_dir} 中未找到 model*.pth 权重文件")
    best_epoch, best_fname = max(checkpoints, key=lambda x: x[0])
    best_path = os.path.join(models_dir, best_fname)
    print(f"加载模型：{best_fname}（第 {best_epoch} 轮）")
else:
    print(f"加载模型：model_best.pth")

model.load_state_dict(torch.load(best_path, map_location=device))
model.to(device)
model.eval()


data = pd.read_csv(test_file_path, encoding="utf-8")
data = data[["title", "introduction", "real_class_no"]].dropna(subset=["real_class_no"])
data = data.dropna(subset=["title", "introduction"], how="all")
data["title"]        = data["title"].fillna("").astype(str).str.strip()
data["introduction"] = data["introduction"].fillna("").astype(str).str.strip()
title_list    = data["title"].tolist()
intro_list    = data["introduction"].tolist()
classify_list = data["real_class_no"].tolist()


total_bleu = 0.0
total_num  = 0
all_preds: list[str]  = []
all_labels: list[str] = []

for title, intro, classify in zip(title_list, intro_list, classify_list):
    total_num += 1
    pred_seq  = predict_hierarchical(model, title, intro, level_vocabs, device)
    classify  = str(classify).split("/")[0]

    accuracy_bleu = bleu(pred_seq, classify, k=2)
    total_bleu   += accuracy_bleu
    all_preds.append(pred_seq)
    all_labels.append(classify)

    if total_num % 100 == 0:
        print(f"pred_seq:{pred_seq}")
        print(f"classify:{classify}")
        print(f"BLEU {accuracy_bleu:.3f}")
        print("--------------------------------")

print(f"Avg BLEU {total_bleu / max(total_num, 1):.3f}")
metrics = hierarchical_accuracy(all_preds, all_labels, levels=(1, 2, 3, 4, 5))
print(f'Exact Match: {metrics["exact_match"]:.3f}  (total={metrics["total"]})')
for lvl in (1, 2, 3, 4, 5):
    print(
        f'  level@{lvl}: {metrics[f"level@{lvl}"]:.3f}  (support={metrics[f"level@{lvl}_support"]})'
    )



# # 生成式模型评测代码

# import os
# import torch
# import pandas as pd

# from arch.encoder_decoder import EncoderDecoder
# from arch.Transformer import TransformerDecoder
# from arch.train_pred import predict_bert_seq2seq, beam_search_bert_seq2seq
# from arch.BERT import BERT_encoder
# from utils.Accurancy import bleu, hierarchical_accuracy
# from utils.text_handle import WordCount


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# test_file_path = os.path.join(BASE_DIR, "data_set/book2024.csv")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# models_dir = os.path.join(BASE_DIR, "models")
# bert_vocab_path = os.path.join(models_dir, "bert_vocab.pth")
# if not os.path.exists(bert_vocab_path):
#     raise FileNotFoundError(
#         f"未找到 BERT 词表：{bert_vocab_path}，请先运行 train_BRET.py 训练并保存 bert_vocab.pth"
#     )

# vocab_state = torch.load(bert_vocab_path, map_location="cpu")
# tgt_vocab = WordCount.from_state(vocab_state["tgt_vocab"])
# src_num_steps = vocab_state.get("src_num_steps", 128)
# tgt_num_steps = vocab_state.get("tgt_num_steps", 15)

# # 与 train_BRET.py 中 Decoder / BERT 结构保持一致
# n_heads = 8
# decoder_layers = 4
# dim_feedforward = 1024
# dropout = 0.1


# # 替换成你自己的BERT模型路径
# encoder = BERT_encoder(model_path='E:\\dev\\项目\\AI_Assist_book_cateloging\\model_train\\bert-base-chinese', max_length=src_num_steps, freeze_bert=True)
# decoder = TransformerDecoder(
#     len(tgt_vocab),
#     d_model=768,
#     n_heads=n_heads,
#     decoder_layers=decoder_layers,
#     dim_feedforward=dim_feedforward,
#     dropout=dropout,
# )
# model = EncoderDecoder(encoder, decoder)

# checkpoints = []
# if os.path.isdir(models_dir):
#     for fname in os.listdir(models_dir):
#         if fname.startswith("model") and fname.endswith(".pth"):
#             try:
#                 epoch_num = int(fname[len("model") : -len(".pth")])
#                 checkpoints.append((epoch_num, fname))
#             except ValueError:
#                 pass
# if not checkpoints:
#     raise FileNotFoundError(f"在 {models_dir} 中未找到 model*.pth 权重文件")

# best_epoch, best_fname = max(checkpoints, key=lambda x: x[0])
# ckpt_path = os.path.join(models_dir, best_fname)
# print(f"加载模型：{best_fname}（第 {best_epoch} 轮）")
# model.load_state_dict(torch.load(ckpt_path, map_location=device))
# model.to(device)
# model.eval()

# data = pd.read_csv(test_file_path, encoding="utf-8")
# data = data[["title", "introduction", "real_class_no"]].dropna(subset=["real_class_no"])
# data = data.dropna(subset=["title", "introduction"], how="all")
# data["title"] = data["title"].fillna("").astype(str).str.strip()
# data["introduction"] = data["introduction"].fillna("").astype(str).str.strip()
# title_list = data["title"].tolist()
# intro_list = data["introduction"].tolist()
# classify_list = data["real_class_no"].tolist()

# total_bleu = 0.0
# total_num = 0
# all_preds = []
# all_labels = []

# BEAM_WIDTH = 3   # 设为 1 则退化为贪心解码

# for title, intro, classify in zip(title_list, intro_list, classify_list):
#     total_num += 1
#     if BEAM_WIDTH > 1:
#         pred_seq, _ = beam_search_bert_seq2seq(
#             model, title, intro, tgt_vocab, tgt_num_steps, device=device, beam_width=BEAM_WIDTH
#         )
#     else:
#         pred_seq, _ = predict_bert_seq2seq(
#             model, title, intro, tgt_vocab, tgt_num_steps, device=device
#         )
#     pred_seq = pred_seq.split("<bos>")[-1]
#     classify = str(classify).split("/")[0]
#     accuracy_bleu = bleu(pred_seq, classify, k=2)
#     total_bleu += accuracy_bleu
#     all_preds.append(pred_seq)
#     all_labels.append(classify)
#     if total_num % 100 == 0:
#         print(f"pred_seq:{pred_seq}")
#         print(f"classify:{classify}")
#         print(f"BLEU {accuracy_bleu:.3f}")
#         print("--------------------------------")

# print(f"Avg BLEU {total_bleu / total_num:.3f}")
# metrics = hierarchical_accuracy(all_preds, all_labels, levels=(1, 2, 3, 4, 5))
# print(f'Exact Match: {metrics["exact_match"]:.3f}  (total={metrics["total"]})')
# for lvl in (1, 2, 3, 4, 5):
#     print(
#         f'  level@{lvl}: {metrics[f"level@{lvl}"]:.3f}  (support={metrics[f"level@{lvl}_support"]})'
#     )
