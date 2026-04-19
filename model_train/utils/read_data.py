from torch.utils.data import Dataset
from .text_handle import WordCount, intro_tokenize_batch
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
    data = pd.read_csv(file_path,encoding='utf-8')
    data = data.dropna(subset=['real_class_no'])
    data = data.dropna(subset=['title', 'introduction'], how='all')
    title_list = data['title'].fillna('').astype(str).str.strip()
    introduction_list = data['introduction'].fillna('').astype(str).str.strip()
    intro_list = (
        title_list + ' ' + introduction_list
    ).str.strip().tolist()
    classify_list = data['real_class_no'].tolist()
    intro_token_list = intro_tokenize_batch(intro_list)
    classify_token_list = [list(cls.split('/')[0]) for cls in classify_list]
    return intro_token_list, classify_token_list

class TextDataset(Dataset):
    def __init__(self, intro_token_list, classify_token_list,
                 min_freq=1, max_freq=None, src_num_steps=256, tgt_num_steps=16):
        self.vocab = WordCount(intro_token_list, min_freq=min_freq, max_freq=max_freq,
                               reserved_tokens=['<pad>', '<unk>', '<eos>'])
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
            [bos] + [self.label_vocab[t] for t in tokens if t in self.label_vocab.word_index] + [eos]
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

if __name__ == "__main__":
    intro_token_list, classify_token_list = build_dataset("data_set/book2019-2023.csv")
    print(intro_token_list[0])
    print(classify_token_list[0])