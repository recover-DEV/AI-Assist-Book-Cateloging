"""BERT 层级判别式分类器 —— 用于中图分类法等层级分类号预测。

层级定义（以 "TN929.11" 为例）：
  level=1     "T"
  level=2     "TN"
  level=3     "TN9"
  level=4     "TN92"
  level=5     "TN929"
  level=None  "TN929.11"  （完整分类号）
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset


def build_level_vocab(classify_list: list, level: int | None) -> dict[str, int]:
    """为某一层级构建 str→int 词表。

    level=None  : 使用完整分类号
    level=k(int): 使用分类号前 k 个字符（不足 k 位则使用完整码）
    """
    codes: set[str] = set()
    for code in classify_list:
        code_s = str(code).split("/")[0].strip()
        if level is None:
            codes.add(code_s)
        else:
            codes.add(code_s[:level] if len(code_s) >= level else code_s)
    vocab: dict[str, int] = {"<unk>": 0}
    for c in sorted(codes):
        vocab[c] = len(vocab)
    return vocab


def build_all_level_vocabs(classify_list: list, levels: list) -> list[dict[str, int]]:
    """为 levels 中每个层级构建词表，返回列表（顺序与 levels 一致）。"""
    return [build_level_vocab(classify_list, lv) for lv in levels]


class HierarchicalLabelDataset(Dataset):
    """全量预 tokenize + 各层级标签索引，训练时 DataLoader 直接读取，零重复计算。

    __getitem__ 返回：
        (input_ids, attention_mask, token_type_ids, label_lv0, label_lv1, …, label_full)
    """

    def __init__(
        self,
        encoder,
        title_list: list[str],
        intro_list: list[str],
        classify_list: list,
        level_vocabs: list[dict[str, int]],
        levels: list,
    ):
        tokenize = encoder.tokenize_text(title_list, intro_list)
        self.input_ids      = tokenize["input_ids"].long().contiguous()
        self.attention_mask = tokenize["attention_mask"].long().contiguous()
        tt = tokenize.get("token_type_ids", None)
        self.token_type_ids = (
            tt.long().contiguous() if tt is not None else torch.zeros_like(self.input_ids)
        )

        self.level_labels: list[torch.Tensor] = []
        for level, vocab in zip(levels, level_vocabs):
            unk = vocab.get("<unk>", 0)
            labels = []
            for code in classify_list:
                code_s = str(code).split("/")[0].strip()
                key = (
                    code_s[:level]
                    if (level is not None and len(code_s) >= level)
                    else code_s
                )
                labels.append(vocab.get(key, unk))
            self.level_labels.append(torch.tensor(labels, dtype=torch.long))

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.token_type_ids[idx],
            *[ll[idx] for ll in self.level_labels],
        )


class BERTHierarchicalClassifier(nn.Module):

    def __init__(
        self,
        bert_encoder,
        level_vocab_sizes: list[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = bert_encoder
        self.dropout = nn.Dropout(dropout)
        hidden = self.encoder.bert.config.hidden_size
        self.heads = nn.ModuleList([
            nn.Linear(hidden, n) for n in level_vocab_sizes
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> list[torch.Tensor]:
        out = self.encoder.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_vec = self.dropout(out.last_hidden_state[:, 0, :])  
        return [head(cls_vec) for head in self.heads]          


def predict_hierarchical(
    model: BERTHierarchicalClassifier,
    title: str,
    intro: str,
    level_vocabs: list[dict[str, int]],
    device: torch.device,
) -> str:
    """使用 full code 头（最后一个头）预测完整分类号。"""
    model.eval()
    model.to(device)
    with torch.no_grad():
        tokenize = model.encoder.tokenize_text(title, intro)
        tokenize = {k: v.to(device) for k, v in tokenize.items() if isinstance(v, torch.Tensor)}
        tti = tokenize.get("token_type_ids", torch.zeros_like(tokenize["input_ids"]))
        logits_list = model(tokenize["input_ids"], tokenize["attention_mask"], tti)
    pred_idx = logits_list[-1].argmax(dim=-1).item()
    inv_vocab = {v: k for k, v in level_vocabs[-1].items()}
    return inv_vocab.get(pred_idx, "<unk>")
