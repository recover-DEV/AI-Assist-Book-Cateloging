import os
from utils.read_data import build_dataset, TextDataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data_set/book2019-2023.csv")

intro_token_list, classify_token_list = build_dataset(file_path)
print(intro_token_list[0])
print(classify_token_list[0])
dataset = TextDataset(intro_token_list, classify_token_list, min_freq=1)

print(dataset[0])
