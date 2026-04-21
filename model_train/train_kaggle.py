
from __future__ import annotations

import collections
import math
import os
import sys
import warnings
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertModel, BertTokenizer

# -----------------------------------------------------------------------------
# 路径（不依赖 sys.path 注入本地包）
# Jupyter / exec 中无 __file__，用 BERT_BASE_DIR 或当前工作目录。
# -----------------------------------------------------------------------------
_script_path = globals().get("__file__")
if _script_path is not None:
    BASE_DIR = os.path.dirname(os.path.abspath(_script_path))
else:
    BASE_DIR = os.path.abspath(os.environ.get("BERT_BASE_DIR", os.getcwd()))
os.chdir(BASE_DIR)

# Kaggle 默认路径（与 Add Data 挂载一致）；本地不存在则自动回退，可用环境变量覆盖。
KAGGLE_DATA_DIR = "/kaggle/input/datasets/recoverpeng/data-set"
KAGGLE_BERT_PRETRAINED = "/kaggle/input/models/recoverpeng/bert-base-chinese/pytorch/default/1"

# =============================================================================
# utils/text_handle — WordCount（训练 BERT 标签词表仅需此类）
# =============================================================================


class WordCount:
    def __init__(self, text_list, min_freq=0, max_freq=None, reserved_tokens=None):
        self.text_list = text_list
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.reserved_tokens = reserved_tokens or []
        self.word_count: dict[str, int] = {}
        self.word_index: dict[str, int] = {}
        self.index_word: dict[int, str] = {}
        self.count_word()
        self.build_word_index()

    def count_word(self):
        for text in self.text_list:
            for token in text:
                self.word_count[token] = self.word_count.get(token, 0) + 1
        return self.word_count

    def build_word_index(self):
        self.index_word = {}
        for token in self.reserved_tokens:
            idx = len(self.word_index)
            self.word_index[token] = idx
            self.index_word[idx] = token
        for word, count in self.word_count.items():
            if count < self.min_freq:
                continue
            if self.max_freq is not None and count > self.max_freq:
                continue
            if word not in self.word_index:
                idx = len(self.word_index)
                self.word_index[word] = idx
                self.index_word[idx] = word
        return self.word_index

    def __getitem__(self, query):
        if isinstance(query, int):
            return self.index_word[query]
        return self.word_index[query]

    def __len__(self):
        return len(self.word_index)

    def to_state(self):
        return {
            "min_freq": self.min_freq,
            "max_freq": self.max_freq,
            "reserved_tokens": self.reserved_tokens,
            "word_count": self.word_count,
            "word_index": self.word_index,
        }

    @classmethod
    def from_state(cls, state):
        vocab = cls(
            text_list=[],
            min_freq=state.get("min_freq", 0),
            max_freq=state.get("max_freq"),
            reserved_tokens=state.get("reserved_tokens", []),
        )
        vocab.word_count = dict(state.get("word_count", {}))
        vocab.word_index = dict(state["word_index"])
        vocab.index_word = {idx: tok for tok, idx in vocab.word_index.items()}
        return vocab


# =============================================================================
# utils/read_data — BERT 数据
# =============================================================================


def truncate_pad(indices, num_steps, pad_idx):
    if len(indices) > num_steps:
        return indices[:num_steps]
    return indices + [pad_idx] * (num_steps - len(indices))


def build_bert_dataset(file_path):
    data = pd.read_csv(file_path, encoding="utf-8")
    data = data.dropna(subset=["real_class_no"])
    data = data.dropna(subset=["title", "introduction"], how="all")
    title_list = data["title"].fillna("").astype(str).str.strip().tolist()
    introduction_list = data["introduction"].fillna("").astype(str).str.strip().tolist()
    classify_list = data["real_class_no"].tolist()
    return title_list, introduction_list, classify_list


class BERTTextDataset(Dataset):
    def __init__(self, encoder, title_list, intro_list, classify_token_list, min_freq=0, tgt_num_steps=16):
        if not classify_token_list:
            raise ValueError("classify_token_list is empty")
        if isinstance(classify_token_list[0], str):
            classify_token_list = [list(str(c).split("/")[0]) for c in classify_token_list]

        self.label_vocab = WordCount(
            classify_token_list,
            min_freq=min_freq,
            reserved_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        )

        tokenize = encoder.tokenize_text(title_list, intro_list)
        self.input_ids = tokenize["input_ids"].long().contiguous()
        self.attention_mask = tokenize["attention_mask"].long().contiguous()
        if "token_type_ids" in tokenize and tokenize["token_type_ids"] is not None:
            self.token_type_ids = tokenize["token_type_ids"].long().contiguous()
        else:
            self.token_type_ids = None

        self.valid_x_lens = self.attention_mask.sum(dim=1).tolist()

        tgt_pad = self.label_vocab["<pad>"]
        eos = self.label_vocab["<eos>"]
        tgt_indices = [
            [self.label_vocab[t] for t in tokens if t in self.label_vocab.word_index] + [eos]
            for tokens in classify_token_list
        ]
        self.valid_y_lens = [torch.tensor(min(len(s), tgt_num_steps)) for s in tgt_indices]
        self.labels = [torch.tensor(truncate_pad(s, tgt_num_steps, tgt_pad)) for s in tgt_indices]

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        if self.token_type_ids is not None:
            tt = self.token_type_ids[idx]
        else:
            tt = torch.zeros_like(self.input_ids[idx])
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            tt,
            torch.tensor(self.valid_x_lens[idx], dtype=torch.long),
            self.labels[idx],
            self.valid_y_lens[idx],
        )


# =============================================================================
# arch/encoder_decoder
# =============================================================================


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args, tgt_key_padding_mask=None):
        enc_outputs = self.encoder(enc_X, *args)
        if isinstance(enc_outputs, dict) and "last_hidden_state" in enc_outputs:
            memory = enc_outputs["last_hidden_state"]
        else:
            memory = getattr(enc_outputs, "last_hidden_state", enc_outputs)
        dec_state = self.decoder.init_state(memory, *args)
        return self.decoder(dec_X, dec_state, tgt_key_padding_mask=tgt_key_padding_mask)


# =============================================================================
# arch/Transformer — Decoder 子集
# =============================================================================


def generate_tgt_mask(T, device):
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


class TransformerDecoder(Decoder):
    def __init__(self, tgt_vocab_size, d_model, n_heads, decoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        transformer_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(transformer_layer, decoder_layers)
        self.dense = nn.Linear(d_model, tgt_vocab_size)

    def init_state(self, memory, enc_valid_len, *args):
        enc_mask = torch.arange(memory.shape[1], device=memory.device).unsqueeze(0) >= enc_valid_len.unsqueeze(1)
        return [memory, enc_mask]

    def forward(self, tgt, state, tgt_key_padding_mask=None):
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt * math.sqrt(self.d_model))
        memory, enc_valid_len = state
        tgt_mask = generate_tgt_mask(tgt.shape[1], device=tgt.device)
        tgt = self.transformer_decoder(
            tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=enc_valid_len,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        tgt = self.dense(tgt)
        return tgt, state


# =============================================================================
# arch/BERT
# =============================================================================


class BERT_encoder(Encoder):
    def __init__(self, model_path=None, max_length=256, freeze_bert=False):
        super().__init__()
        self.max_length = max_length
        if model_path is not None:
            resolved = model_path
        elif os.environ.get("BERT_PRETRAINED_PATH"):
            resolved = os.environ["BERT_PRETRAINED_PATH"]
        elif os.path.isdir(KAGGLE_BERT_PRETRAINED):
            resolved = KAGGLE_BERT_PRETRAINED
        else:
            resolved = "bert-base-chinese"
        self._pretrained_path = resolved
        self.bert = BertModel.from_pretrained(resolved)
        for i in range(6):
            for p in self.bert.encoder.layer[i].parameters():
                p.requires_grad = False
        if freeze_bert:
            for p in self.bert.embeddings.parameters():
                p.requires_grad = False
        self.bert_tokenizer = BertTokenizer.from_pretrained(resolved)

    def tokenize_text(self, title_list, intro_list):
        if isinstance(title_list, str):
            title_list = [title_list]
        if isinstance(intro_list, str):
            intro_list = [intro_list]
        return self.bert_tokenizer(
            title_list,
            intro_list,
            padding=True,
            truncation="only_second",
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(self, x, *args, **kwargs):
        if not isinstance(x, dict):
            raise TypeError("BERT_encoder expects tokenizer output dict")
        return self.bert(**x)


# =============================================================================
# arch/train_pred — BERT 训练 / 预测
# =============================================================================


def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange(maxlen, dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = "none"
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def grad_clipping(net, theta):
    params = [p for p in net.parameters() if p.requires_grad and p.grad is not None]
    norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in params))
    if norm > theta:
        for param in params:
            param.grad.data *= theta / norm
    return norm


def _xla_step_boundary() -> None:
    """XLA 在每个 step 末尾需要一次「边界」以提交图；新版推荐 torch_xla.sync()，旧版为 xm.mark_step()。"""
    try:
        import torch_xla as tx

        sync_fn = getattr(tx, "sync", None)
        if callable(sync_fn):
            sync_fn()
            return
    except Exception:
        pass
    if _XM is not None and hasattr(_XM, "mark_step"):
        _XM.mark_step()


def _eval_loss(net, val_iter, loss_fn, tgt_vocab, device, is_bert_encoder=False, use_amp=False):
    net.eval()
    total_loss, total_tokens = 0.0, 0
    amp_enabled = use_amp and getattr(device, "type", "") == "cuda"
    with torch.no_grad():
        for batch in val_iter:
            if is_bert_encoder:
                input_ids, attention_mask, token_type_ids, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
                enc_X = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                }
            else:
                X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
                enc_X = X
            bos = torch.tensor([tgt_vocab["<bos>"]] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            tgt_key_padding_mask = torch.arange(dec_input.shape[1], device=device).unsqueeze(0) >= Y_valid_len.long().unsqueeze(1)
            with autocast(device_type="cuda", enabled=amp_enabled):
                Y_hat, _ = net(enc_X, dec_input, X_valid_len, tgt_key_padding_mask=tgt_key_padding_mask)
            l = loss_fn(Y_hat.float(), Y, Y_valid_len)
            total_loss += float(l.sum().detach().cpu())
            total_tokens += int(Y_valid_len.sum().detach().cpu().item())
    net.train()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def train_bert_seq2seq(
    net,
    data_iter,
    optimizer,
    num_epochs,
    tgt_vocab,
    device,
    save_dir,
    save_epoch,
    eval_epoch,
    start_epoch=0,
    val_iter=None,
    scheduler=None,
    use_amp=None,
):
    if start_epoch == 0:

        def xavier_init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
            if type(m) == nn.GRU:
                for param in m._flat_weights_names:
                    if "weight" in param:
                        nn.init.xavier_uniform_(m._parameters[param])

        net.decoder.apply(xavier_init_weights)
    net.to(device)

    if use_amp is None:
        use_amp = getattr(device, "type", "") == "cuda"
    amp_enabled = bool(use_amp) and getattr(device, "type", "") == "cuda"
    scaler = None if IS_XLA else GradScaler("cuda", enabled=amp_enabled)

    loss = MaskedSoftmaxCELoss()
    best_val_loss = float("inf")
    best_model_path = None

    net.train()
    for epoch in range(num_epochs):
        epoch_loss_sum = 0.0
        epoch_n_samples = 0
        for batch in data_iter:
            optimizer.zero_grad()
            input_ids, attention_mask, token_type_ids, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            enc_X = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
            bos = torch.tensor([tgt_vocab["<bos>"]] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            tgt_key_padding_mask = torch.arange(dec_input.shape[1], device=device).unsqueeze(0) >= Y_valid_len.long().unsqueeze(1)
            if IS_XLA:
                Y_hat, _ = net(enc_X, dec_input, X_valid_len, tgt_key_padding_mask=tgt_key_padding_mask)
                l = loss(Y_hat.float(), Y, Y_valid_len)
                epoch_loss_sum += float(l.sum().detach().cpu())
                epoch_n_samples += int(Y.size(0))
                l.sum().backward()
                grad_clipping(net, 1)
                _XM.optimizer_step(optimizer)
                _xla_step_boundary()
            else:
                with autocast(device_type="cuda", enabled=amp_enabled):
                    Y_hat, _ = net(enc_X, dec_input, X_valid_len, tgt_key_padding_mask=tgt_key_padding_mask)
                l = loss(Y_hat.float(), Y, Y_valid_len)
                epoch_loss_sum += float(l.sum().detach().cpu())
                epoch_n_samples += int(Y.size(0))
                assert scaler is not None
                scaler.scale(l.sum()).backward()
                scaler.unscale_(optimizer)
                grad_clipping(net, 1)
                scaler.step(optimizer)
                scaler.update()

        if (epoch + 1) % save_epoch == 0:
            path = os.path.join(save_dir, "model" + str(epoch + start_epoch + 1) + ".pth")
            _save_model_state_cpu(net, path)
            print(f"model saved to {path}, train epoch {start_epoch + epoch + 1}")

        if (epoch + 1) % eval_epoch == 0:
            train_loss = epoch_loss_sum / max(epoch_n_samples, 1)
            log_msg = f"epoch {epoch + 1}, train_loss {train_loss:.9f}"
            if val_iter is not None:
                val_loss = _eval_loss(net, val_iter, loss, tgt_vocab, device, is_bert_encoder=True, use_amp=amp_enabled)
                log_msg += f", val_loss {val_loss:.9f}"
                if scheduler is not None:
                    scheduler.step(val_loss)
                    log_msg += f', lr {optimizer.param_groups[0]["lr"]:.2e}'
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(save_dir, "model_best.pth")
                    _save_model_state_cpu(net, best_model_path)
                    log_msg += "  ← best"
            print(log_msg)

    if best_model_path:
        print(f"最佳模型已保存至 {best_model_path}，验证 loss={best_val_loss:.9f}")


def predict_bert_seq2seq(net, title, intro, tgt_vocab, num_steps, device):
    net.eval()
    net.to(device)
    tokenize = net.encoder.tokenize_text(title, intro)
    tokenize = {k: tokenize[k].to(device) for k in tokenize if isinstance(tokenize[k], torch.Tensor)}
    feature = net.encoder(tokenize)["last_hidden_state"]
    enc_valid_len = tokenize["attention_mask"].sum(dim=1).to(device)
    dec_state = net.decoder.init_state(feature, enc_valid_len)
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab["<bos>"]], dtype=torch.long, device=device), dim=0)
    output_seq = []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        next_token = Y[:, -1:, :].argmax(dim=2)
        pred = next_token.squeeze().type(torch.int32).item()
        if pred == tgt_vocab["<eos>"]:
            break
        output_seq.append(pred)
        dec_X = torch.cat([dec_X, next_token], dim=1)
    return "".join(tgt_vocab.index_word.get(i, "<unk>") for i in output_seq), 1


# =============================================================================
# utils/Accurancy — 评测指标
# =============================================================================


def bleu(pred_seq, label_seq, k):
    pred_tokens, label_tokens = list(pred_seq), list(label_seq)
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    if len_pred == 0:
        return 0.0
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs["".join(label_tokens[i : i + n])] += 1
        denom = len_pred - n + 1
        if denom <= 0:
            label_counter = collections.Counter(label_tokens)
            pred_counter = collections.Counter(pred_tokens)
            char_matches = sum(min(cnt, label_counter[tok]) for tok, cnt in pred_counter.items())
            return char_matches / len_label
        for i in range(denom):
            key = "".join(pred_tokens[i : i + n])
            if label_subs[key] > 0:
                num_matches += 1
                label_subs[key] -= 1
        score *= math.pow(num_matches / denom, math.pow(0.5, n))
    return score


def normalize_class_no(class_no):
    if class_no is None:
        return ""
    code = str(class_no).strip()
    if "/" in code:
        code = code.split("/")[0]
    return code


def hierarchical_accuracy(pred_list, label_list, levels=(1, 2, 3, 4)):
    pred_len = len(pred_list)
    label_len = len(label_list)
    paired_len = min(pred_len, label_len)
    levels = sorted(set(int(l) for l in levels if int(l) > 0))
    total = 0
    exact_hits = 0
    level_hits = {level: 0 for level in levels}
    level_totals = {level: 0 for level in levels}

    for pred, label in zip(pred_list[:paired_len], label_list[:paired_len]):
        pred_code = normalize_class_no(pred)
        label_code = normalize_class_no(label)
        if not label_code:
            continue
        total += 1
        if pred_code == label_code:
            exact_hits += 1
        for level in levels:
            if len(label_code) < level:
                continue
            level_totals[level] += 1
            gold_prefix = label_code[:level]
            pred_prefix = pred_code[:level]
            if len(pred_code) >= level and pred_prefix == gold_prefix:
                level_hits[level] += 1

    if total == 0:
        result = {
            "total": 0,
            "exact_match": 0.0,
            "paired_samples": paired_len,
            "dropped_pred": max(0, pred_len - paired_len),
            "dropped_label": max(0, label_len - paired_len),
        }
        for level in levels:
            result[f"level@{level}"] = 0.0
            result[f"level@{level}_support"] = 0
        return result

    result = {
        "total": total,
        "exact_match": exact_hits / total,
        "paired_samples": paired_len,
        "dropped_pred": max(0, pred_len - paired_len),
        "dropped_label": max(0, label_len - paired_len),
    }
    for level in levels:
        support = level_totals[level]
        result[f"level@{level}"] = (level_hits[level] / support) if support > 0 else 0.0
        result[f"level@{level}_support"] = support
    return result


# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
TRAIN_CSV = "book2019-2023.csv"
EVAL_CSV = "book2024.csv"
batch_size = 64
src_num_steps = 128
tgt_num_steps = 15
lr = 3e-4
num_epochs = 20
val_ratio = 0.05
save_epoch = 1
eval_epoch = 1

n_heads = 8
decoder_layers = 2
dim_feedforward = 512
dropout = 0.3

# -----------------------------------------------------------------------------
# 设备：CUDA >（可选）XLA/TPU > CPU。BERT_DEVICE=xla|tpu|cuda|gpu|cpu 可强制。
# -----------------------------------------------------------------------------
IS_XLA = False
_XM = None  # type: ignore[assignment]
_PL = None  # type: ignore[assignment]


def _kaggle_xla_single_process_env() -> None:
    """
    Kaggle 等环境会预置 TPU_PROCESS_ADDRESSES，与单进程 xm.xla_device() / PJRT 冲突，
    触发：Could not find SliceBuilder port … `tpu_process_addresses`="local"（0 ports）。
    见 Kaggle 讨论与 pytorch-lightning #20244 / pytorch/xla #5616。
    """
    if os.environ.get("BERT_DISABLE_KAGGLE_XLA_HACK") == "1":
        return
    for key in ("TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID"):
        os.environ.pop(key, None)
    if os.path.isdir("/kaggle/working"):
        os.environ.setdefault("PJRT_DEVICE", "TPU")


def _try_init_xla() -> torch.device:
    """初始化 torch-xla，设置全局 _XM / _PL / IS_XLA，返回 xla device。"""
    global IS_XLA, _XM, _PL
    # 必须在 import torch_xla 之前清理，否则运行时可能已读到错误配置
    _kaggle_xla_single_process_env()
    # Kaggle 镜像里常见、与训练逻辑无关；设 BERT_SHOW_XLA_WARN=1 可恢复显示
    if os.environ.get("BERT_SHOW_XLA_WARN") != "1":
        warnings.filterwarnings("ignore", category=UserWarning, module="torch_xla")
        warnings.filterwarnings("ignore", category=UserWarning, module="jax._src.cloud_tpu_init")
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

    _XM = xm
    _PL = pl
    IS_XLA = True
    import torch_xla as xla_root

    if hasattr(xla_root, "device"):
        return xla_root.device()
    return xm.xla_device()


def _resolve_training_device() -> torch.device:
    global IS_XLA, _XM, _PL
    IS_XLA = False
    _XM = _PL = None
    pref = (os.environ.get("BERT_DEVICE") or "").strip().lower()
    if pref in ("cpu",):
        return torch.device("cpu")
    if pref in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("BERT_DEVICE=cuda/gpu 但当前环境没有可用 CUDA")
        return torch.device("cuda")
    if pref in ("xla", "tpu"):
        return _try_init_xla()
    # 自动：优先 CUDA，否则尝试 XLA（Kaggle TPU 通常无 CUDA）
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        return _try_init_xla()
    except Exception as e:
        if os.path.isdir("/kaggle/working"):
            print(f"[XLA] 初始化失败，已退回 CPU（TPU 不参与）: {e!r}")
        return torch.device("cpu")


def _save_model_state_cpu(model: nn.Module, path: str) -> None:
    """XLA/CUDA 通用：保存到 CPU，避免 xla 张量直接 pickle 到文件。"""
    sd = model.state_dict()
    cpu_sd = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in sd.items()}
    torch.save(cpu_sd, path)


# 延迟解析 device，避免 import 本模块时过早落到 CPU（Notebook 可先设 BERT_DEVICE 再 main）。
device: torch.device | None = None


def ensure_device() -> torch.device:
    global device
    if device is not None:
        return device
    device = _resolve_training_device()
    return device


def _warn_kaggle_tpu_session_on_cpu() -> None:
    if device is None or not os.path.isdir("/kaggle/working"):
        return
    if getattr(device, "type", "") != "cpu":
        return
    if torch.cuda.is_available():
        return
    print(
        "\n[Kaggle] 当前 device=CPU。若已选 TPU，请在调用 main() 前设置 "
        "os.environ['BERT_DEVICE']='xla'，并查看上方 [XLA] 初始化失败 信息。\n"
    )


def _default_bert_data_dir() -> str:
    if os.environ.get("BERT_DATA_DIR"):
        return os.environ["BERT_DATA_DIR"]
    train_name = TRAIN_CSV
    if os.path.isfile(os.path.join(KAGGLE_DATA_DIR, train_name)):
        return KAGGLE_DATA_DIR
    return os.path.join(BASE_DIR, "data_set")


DATA_DIR = _default_bert_data_dir()
train_file_path = os.path.join(DATA_DIR, TRAIN_CSV)
eval_file_path = os.path.join(DATA_DIR, EVAL_CSV)

if os.path.isdir("/kaggle/working"):
    save_dir = os.path.join("/kaggle/working", "models")
else:
    save_dir = os.path.join(BASE_DIR, "models")
os.makedirs(save_dir, exist_ok=True)


def _build_model(tgt_vocab: WordCount, max_len: int) -> EncoderDecoder:
    encoder = BERT_encoder(max_length=max_len, freeze_bert=True)
    decoder = TransformerDecoder(
        len(tgt_vocab),
        d_model=768,
        n_heads=n_heads,
        decoder_layers=decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    return EncoderDecoder(encoder, decoder)


def _load_checkpoint(model: EncoderDecoder, save_dir: str, device: torch.device) -> int:
    checkpoints: list[tuple[int, str]] = []
    if os.path.isdir(save_dir):
        for fname in os.listdir(save_dir):
            if fname.startswith("model") and fname.endswith(".pth") and fname != "model_best.pth":
                try:
                    epoch_num = int(fname[len("model") : -len(".pth")])
                    checkpoints.append((epoch_num, fname))
                except ValueError:
                    pass
    if not checkpoints:
        return 0
    best_epoch, best_fname = max(checkpoints, key=lambda x: x[0])
    ckpt_path = os.path.join(save_dir, best_fname)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device)
    print(f"已加载模型 {best_fname}，从第 {best_epoch} 轮继续训练")
    return best_epoch


def run_eval_on_csv(model: EncoderDecoder, tgt_vocab: WordCount, csv_path: str, tgt_ns: int) -> None:
    if not os.path.isfile(csv_path):
        print(f"跳过评测：未找到 {csv_path}")
        return
    ensure_device()
    assert device is not None
    model.to(device)
    model.eval()
    data = pd.read_csv(csv_path, encoding="utf-8")
    data = data[["title", "introduction", "real_class_no"]].dropna(subset=["real_class_no"])
    data = data.dropna(subset=["title", "introduction"], how="all")
    data["title"] = data["title"].fillna("").astype(str).str.strip()
    data["introduction"] = data["introduction"].fillna("").astype(str).str.strip()
    title_list = data["title"].tolist()
    intro_list = data["introduction"].tolist()
    classify_list = data["real_class_no"].tolist()

    total_bleu = 0.0
    total_num = 0
    all_preds: list[str] = []
    all_labels: list[str] = []
    for title, intro, classify in zip(title_list, intro_list, classify_list):
        total_num += 1
        pred_seq, _ = predict_bert_seq2seq(model, title, intro, tgt_vocab, tgt_ns, device=device)
        pred_seq = pred_seq.split("<bos>")[-1]
        classify_s = str(classify).split("/")[0]
        accuracy_bleu = bleu(pred_seq, classify_s, k=2)
        total_bleu += accuracy_bleu
        all_preds.append(pred_seq)
        all_labels.append(classify_s)
        if total_num % 100 == 0:
            print(f"pred_seq:{pred_seq}")
            print(f"classify:{classify_s}")
            print(f"BLEU {accuracy_bleu:.3f}")
            print("--------------------------------")
    print(f"Avg BLEU {total_bleu / max(total_num, 1):.3f}")
    metrics = hierarchical_accuracy(all_preds, all_labels, levels=(1, 2, 3, 4, 5))
    print(f'Exact Match: {metrics["exact_match"]:.3f}  (total={metrics["total"]})')
    for lvl in (1, 2, 3, 4, 5):
        print(f'  level@{lvl}: {metrics[f"level@{lvl}"]:.3f}  (support={metrics[f"level@{lvl}_support"]})')


def eval_only_main() -> None:
    ensure_device()
    _warn_kaggle_tpu_session_on_cpu()
    bert_vocab_path = os.path.join(save_dir, "bert_vocab.pth")
    if not os.path.isfile(bert_vocab_path):
        raise FileNotFoundError(f"未找到词表：{bert_vocab_path}")
    vocab_state = torch.load(bert_vocab_path, map_location="cpu")
    tgt_vocab = WordCount.from_state(vocab_state["tgt_vocab"])
    src_ns = vocab_state.get("src_num_steps", src_num_steps)
    tgt_ns = vocab_state.get("tgt_num_steps", tgt_num_steps)
    model = _build_model(tgt_vocab, src_ns)
    ep = _load_checkpoint(model, save_dir, device)
    if ep == 0:
        raise FileNotFoundError(f"在 {save_dir} 中未找到可加载的 model<N>.pth（评测需要已训练权重）")
    run_eval_on_csv(model, tgt_vocab, eval_file_path, tgt_ns)


def main() -> None:
    ensure_device()
    _warn_kaggle_tpu_session_on_cpu()
    assert device is not None
    if not os.path.isfile(train_file_path):
        raise FileNotFoundError(
            f"未找到训练 CSV：{train_file_path}\n"
            f"请设置 BERT_DATA_DIR，或确认 Kaggle 已挂载：{KAGGLE_DATA_DIR}/{TRAIN_CSV}"
        )

    print(f"BASE_DIR={BASE_DIR}")
    print(f"train_file_path={train_file_path}")
    print(f"eval_file_path={eval_file_path}")
    print(f"save_dir={save_dir}")
    print(f"device={device}  IS_XLA={IS_XLA}")
    if os.environ.get("BERT_PRETRAINED_PATH"):
        _bp = os.environ["BERT_PRETRAINED_PATH"]
    elif os.path.isdir(KAGGLE_BERT_PRETRAINED):
        _bp = KAGGLE_BERT_PRETRAINED
    else:
        _bp = "bert-base-chinese (Hub)"
    print(f"BERT_PRETRAINED={_bp}")


    title_list, intro_list, classify_list = build_bert_dataset(train_file_path)
    encoder = BERT_encoder(max_length=src_num_steps, freeze_bert=True)
    bert_dataset = BERTTextDataset(
        encoder,
        title_list,
        intro_list,
        classify_list,
        min_freq=0,
        tgt_num_steps=tgt_num_steps,
    )

    train_data, val_data = random_split(
        bert_dataset, [1 - val_ratio, val_ratio], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    if IS_XLA and _PL is not None:
        train_loader = _PL.MpDeviceLoader(train_loader, device)
        val_loader = _PL.MpDeviceLoader(val_loader, device)

    model = _build_model(bert_dataset.label_vocab, src_num_steps)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    bert_vocab_path = os.path.join(save_dir, "bert_vocab.pth")
    torch.save(
        {
            "tgt_vocab": bert_dataset.label_vocab.to_state(),
            "src_num_steps": src_num_steps,
            "tgt_num_steps": tgt_num_steps,
        },
        bert_vocab_path,
    )
    print(f"已保存词表到 {bert_vocab_path}")

    start_epoch = _load_checkpoint(model, save_dir, device)
    if start_epoch == 0:
        print("未找到已有 model<N>.pth，从第 0 轮开始训练")

    train_bert_seq2seq(
        model,
        train_loader,
        optimizer,
        num_epochs,
        bert_dataset.label_vocab,
        device,
        save_dir,
        save_epoch,
        eval_epoch,
        start_epoch=start_epoch,
        val_iter=val_loader,
        scheduler=scheduler,
        use_amp=None,
    )

    pred_seq, _ = predict_bert_seq2seq(
        model,
        "DWDM系统测试",
        "本书介绍DWDM的发展与起源、系统的网元级测试、系统测试结果举例等。",
        bert_dataset.label_vocab,
        tgt_num_steps,
        device=device,
    )
    pred_seq = pred_seq.split("<bos>")[-1]
    label_seq = "TN929.11"
    print("pred_seq: ", pred_seq)
    print("label_seq: ", label_seq)
    print(f"BLEU {bleu(pred_seq, label_seq, k=3):.3f}")

    if "--eval" in sys.argv:
        run_eval_on_csv(model, bert_dataset.label_vocab, eval_file_path, tgt_num_steps)


if __name__ == "__main__":
    # 勿在此强制 BERT_DEVICE=xla：本机无 TPU 时会直接报错。Kaggle 需 TPU 时在 Notebook 首格自行设置。
    if "--eval-only" in sys.argv:
        eval_only_main()
    else:
        main()