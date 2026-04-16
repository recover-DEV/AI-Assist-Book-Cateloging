from torch.utils.data import Dataset
from text_handle import WordCount, tokenize_text, intro_tokenize_text, intro_tokenize_batch
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def read_data(file_path):
    data = pd.read_csv(os.path.join(BASE_DIR, file_path))
    data = data[['introduction', 'refer_class_no']].dropna()
    return data

def build_dataset(file_path):
    data = pd.read_csv(os.path.join(BASE_DIR, file_path))
    data = data.dropna(subset=['introduction', 'refer_class_no'])
    intro_list = data['introduction'].tolist()
    classify_list = data['refer_class_no'].tolist()
    intro_token_list = intro_tokenize_batch(intro_list)
    classify_token_list = [list(cls) for cls in classify_list]
    return intro_token_list, classify_token_list

def build_dataset(file_path):
    data = pd.read_csv(os.path.join(BASE_DIR, file_path))
    data = data.dropna(subset=['introduction', 'refer_class_no'])
    intro_list = data['introduction'].tolist()
    classify_list = data['refer_class_no'].tolist()
    intro_token_list = intro_tokenize_batch(intro_list)
    classify_token_list = [list(cls) for cls in classify_list]
    return intro_token_list, classify_token_list

class TextDataset(Dataset):
    def __init__(self, intro_token_list, classify_token_list, min_freq=1):
        self.vocab = WordCount(intro_token_list, min_freq=min_freq)
        self.label_vocab = WordCount(classify_token_list, min_freq=min_freq)
        self.X = [[self.vocab[t] for t in tokens if t in self.vocab.word_index]
                  for tokens in intro_token_list]
        self.y = [[self.label_vocab[t] for t in tokens if t in self.label_vocab.word_index]
                  for tokens in classify_token_list]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

if __name__ == "__main__":
    intro_token_list, classify_token_list = build_dataset("data_set/book2019-2023.csv")
    print(intro_token_list[0])
    print(classify_token_list[0])