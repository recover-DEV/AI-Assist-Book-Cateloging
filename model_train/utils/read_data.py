from torch.utils.data import Dataset
from .text_handle import WordCount, intro_tokenize_batch, intro_tokenize_batch_with_sep
import pandas as pd
import torch


def truncate_pad(indices, num_steps, pad_idx):
    """截断或填充序列到固定长度 num_steps。"""
    if len(indices) > num_steps:
        return indices[:num_steps]
    return indices + [pad_idx] * (num_steps - len(indices))

def read_data(file_path):
    data = pd.read_csv(file_path,encoding='utf-8')
    data = data[['title', 'introduction', 'real_class_no']]
    data = data.dropna(subset=['real_class_no'])
    data = data.dropna(subset=['title', 'introduction'], how='all')
    return data

def build_dataset(file_path):
    data = pd.read_csv(file_path, encoding='utf-8')
    data = data.dropna(subset=['real_class_no'])
    data = data.dropna(subset=['title', 'introduction'], how='all')
    title_list        = data['title'].fillna('').astype(str).str.strip().tolist()
    introduction_list = data['introduction'].fillna('').astype(str).str.strip().tolist()
    classify_list     = data['real_class_no'].tolist()
    intro_token_list  = intro_tokenize_batch_with_sep(title_list, introduction_list)
    classify_token_list = [list(cls.split('/')[0]) for cls in classify_list]
    return intro_token_list, classify_token_list

def build_bert_dataset(file_path):
    data = pd.read_csv(file_path, encoding='utf-8')
    data = data.dropna(subset=['real_class_no'])
    data = data.dropna(subset=['title', 'introduction'], how='all')
    title_list        = data['title'].fillna('').astype(str).str.strip().tolist()
    introduction_list = data['introduction'].fillna('').astype(str).str.strip().tolist()
    classify_list     = data['real_class_no'].tolist()
    return title_list, introduction_list, classify_list

class TextDataset(Dataset):
    def __init__(self, intro_token_list, classify_token_list,
                 min_freq=1, max_freq=None, src_num_steps=256, tgt_num_steps=16):
        self.vocab = WordCount(intro_token_list, min_freq=min_freq, max_freq=max_freq,
                               reserved_tokens=['<pad>', '<unk>', '<eos>', '<sep>'])
        self.label_vocab = WordCount(classify_token_list, min_freq=min_freq,
                                     reserved_tokens=['<pad>', '<bos>', '<eos>', '<unk>'])

        src_pad = self.vocab['<pad>']
        tgt_pad = self.label_vocab['<pad>']
        bos = self.label_vocab['<bos>']
        eos = self.label_vocab['<eos>']

        unk = self.vocab['<unk>']
        src_indices = [
            [self.vocab[t] if t in self.vocab.word_index else unk for t in tokens]
            for tokens in intro_token_list
        ]
        tgt_indices = [
            [self.label_vocab[t] for t in tokens if t in self.label_vocab.word_index] + [eos]
            for tokens in classify_token_list
        ]

        self.valid_x_lens = [torch.tensor(min(len(s), src_num_steps)) for s in src_indices]
        self.valid_y_lens = [torch.tensor(min(len(s), tgt_num_steps)) for s in tgt_indices]

        self.features = [torch.tensor(truncate_pad(s, src_num_steps, src_pad)) for s in src_indices]
        self.labels   = [torch.tensor(truncate_pad(s, tgt_num_steps, tgt_pad)) for s in tgt_indices]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.valid_x_lens[idx], self.labels[idx], self.valid_y_lens[idx]

class BERTTextDataset(Dataset):
    def __init__(self, encoder, title_list, intro_list, classify_token_list, min_freq=0, tgt_num_steps=16):
        if not classify_token_list:
            raise ValueError('classify_token_list is empty')
        if isinstance(classify_token_list[0], str):
            classify_token_list = [list(str(c).split('/')[0]) for c in classify_token_list]

        self.label_vocab = WordCount(classify_token_list, min_freq=min_freq,
                                     reserved_tokens=['<pad>', '<bos>', '<eos>', '<unk>'])

        tokenize = encoder.tokenize_text(title_list, intro_list)
        self.input_ids = tokenize['input_ids'].long().contiguous()
        self.attention_mask = tokenize['attention_mask'].long().contiguous()
        if 'token_type_ids' in tokenize and tokenize['token_type_ids'] is not None:
            self.token_type_ids = tokenize['token_type_ids'].long().contiguous()
        else:
            self.token_type_ids = None

        self.valid_x_lens = self.attention_mask.sum(dim=1).tolist()

        tgt_pad = self.label_vocab['<pad>']
        eos = self.label_vocab['<eos>']

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

if __name__ == "__main__":
    intro_token_list, classify_token_list = build_dataset("data_set/book2019-2023.csv")
    print(intro_token_list[0])
    print(classify_token_list[0])