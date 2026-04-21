import pandas as pd

files = [
    "./data_set/book2019-2023.csv",
    "./data_set/book2024.csv",
]

df = pd.concat(
    [pd.read_csv(f, usecols=["title", "introduction"], dtype=str) for f in files],
    ignore_index=True,
)

df["title"]        = df["title"].fillna("")
df["introduction"] = df["introduction"].fillna("")
df["combined"]     = df["title"] + df["introduction"]
df["length"]       = df["combined"].str.len()

max_len = df["length"].max()
max_row = df.loc[df["length"].idxmax()]

over_256 = (df["length"] > 256).sum()
ratio    = over_256 / len(df) * 100

print(f"两个数据集合并后共 {len(df)} 条记录")
print(f"title + introduction 最大长度: {max_len}")
print(f"长度 > 256 的记录数: {over_256} / {len(df)}  ({ratio:.2f}%)")
