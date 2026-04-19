import os
import torch
from arch.encoder_decoder import EncoderDecoder
from arch.LSTM import LSTM_encoder, LSTM_decoder
from arch.Transformer import TransformerEncoder, TransformerDecoder
from arch.train_pred import predict_seq2seq, predict_seq2seq_greedy
from utils.Accurancy import bleu, hierarchical_accuracy
from utils.text_handle import WordCount
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
test_file_path = os.path.join(BASE_DIR, "data_set/book2024.csv")
src_num_steps = 50
tgt_num_steps = 15
min_freq = 1          # 过滤低频词（出现少于1次）
max_freq = None       # 高频词上限由停用词表控制，此处不额外限制；如需数值限制可设为文档数的80%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models_dir = os.path.join(BASE_DIR, "models")
vocab_path = os.path.join(models_dir, "vocab.pth")
if not os.path.exists(vocab_path):
    raise FileNotFoundError(f"未找到词表文件：{vocab_path}，请先运行 train.py 训练并保存词表")
vocab_state = torch.load(vocab_path, map_location='cpu')
src_vocab = WordCount.from_state(vocab_state['src_vocab'])
tgt_vocab = WordCount.from_state(vocab_state['tgt_vocab'])
src_num_steps = vocab_state.get('src_num_steps', src_num_steps)
tgt_num_steps = vocab_state.get('tgt_num_steps', tgt_num_steps)
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

# # LSTM模型
# embedding_size = 256
# num_hiddens = 256
# num_layers = 2
# dropout = 0.1
# encoder = LSTM_encoder(src_vocab_size, embedding_size, num_hiddens, num_layers, dropout)
# decoder = LSTM_decoder(tgt_vocab_size, embedding_size, num_hiddens, num_layers, dropout)
# predict = predict_seq2seq

# Transformer模型
d_model = 256
n_heads = 4
encoder_layers = 2
decoder_layers = 2
dim_feedforward = 512
dropout = 0.2
encoder = TransformerEncoder(src_vocab_size, d_model, n_heads, encoder_layers, dim_feedforward, dropout)
decoder = TransformerDecoder(tgt_vocab_size, d_model, n_heads, decoder_layers, dim_feedforward, dropout)
predict = predict_seq2seq_greedy


encoder_decoder = EncoderDecoder(encoder,decoder)
print(encoder_decoder)

checkpoints = []
if os.path.isdir(models_dir):
    for fname in os.listdir(models_dir):
        if fname.startswith("model") and fname.endswith(".pth"):
            try:
                epoch_num = int(fname[len("model"):-len(".pth")])
                checkpoints.append((epoch_num, fname))
            except ValueError:
                pass
if not checkpoints:
    raise FileNotFoundError(f"在 {models_dir} 中未找到任何模型文件")
best_epoch, best_fname = max(checkpoints, key=lambda x: x[0])
ckpt_path = os.path.join(models_dir, best_fname)
print(f"加载模型：{best_fname}（第 {best_epoch} 轮）")
encoder_decoder.load_state_dict(torch.load(ckpt_path, map_location=device))

encoder_decoder.to(device)

encoder_decoder.eval()

data = pd.read_csv(test_file_path)
data = data[['title', 'introduction', 'real_class_no']].dropna(subset=['real_class_no'])
data = data.dropna(subset=['title', 'introduction'], how='all')
data['title']        = data['title'].fillna('').astype(str).str.strip()
data['introduction'] = data['introduction'].fillna('').astype(str).str.strip()
title_list    = data['title'].tolist()
intro_list    = data['introduction'].tolist()
classify_list = data['real_class_no'].tolist()
total_bleu = 0
total_num = 0
all_preds = []
all_labels = []
for title, intro, classify in zip(title_list, intro_list, classify_list):
    total_num += 1
    pred_seq,_ = predict(encoder_decoder, intro, src_vocab, tgt_vocab, src_num_steps, device, title=title)
    pred_seq = pred_seq.split('<bos>')[-1]
    classify = classify.split('/')[0]
    accuracy_bleu = bleu(pred_seq, classify, k=2)
    total_bleu += accuracy_bleu
    all_preds.append(pred_seq)
    all_labels.append(classify)
    if total_num % 100 == 0:
        print(f'pred_seq:{pred_seq}')
        print(f'classify:{classify}')
        print(f'BLEU {accuracy_bleu:.3f}')
        print('--------------------------------')

print(f'Avg BLEU {total_bleu / total_num:.3f}')
metrics = hierarchical_accuracy(all_preds, all_labels, levels=(1, 2, 3, 4, 5))
print(f'Exact Match: {metrics["exact_match"]:.3f}  (total={metrics["total"]})')
for lvl in (1, 2, 3, 4, 5):
    print(f'  level@{lvl}: {metrics[f"level@{lvl}"]:.3f}  (support={metrics[f"level@{lvl}_support"]})')