import hanlp
import re

_tokenizer = None

# 书目简介中普遍出现但对分类无帮助的停用词
STOP_WORDS = {
    '本书', '书', '一书', '全书', '该书',
}

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
    return _tokenizer

def _filter_tokens(tokens):
    return [t for t in tokens
            if re.search(r'[\u4e00-\u9fff\w]', t) and t not in STOP_WORDS]

def intro_tokenize_text(text):
    tokens = _get_tokenizer()(text)
    return _filter_tokens(tokens)

def intro_tokenize_batch(text_list, batch_size=64):
    tokenizer = _get_tokenizer()
    result = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        batch_tokens = tokenizer(batch)
        result.extend([_filter_tokens(tokens) for tokens in batch_tokens])
    return result

def intro_tokenize_batch_with_sep(title_list, intro_list, batch_size=64):
    """分别对 title 和 introduction 分词，中间插入 ['<sep>'] 后合并。"""
    title_tokens = intro_tokenize_batch(title_list, batch_size)
    intro_tokens = intro_tokenize_batch(intro_list, batch_size)
    return [t + ['<sep>'] + i for t, i in zip(title_tokens, intro_tokens)]

def classify_tokenize_text(text):
    text = text.split('/')[0] 
    return list(text)

def tokenize_text(text_list, tokenizer):
    result = list()
    for text in text_list:
        result.append(tokenizer(text))
    return result

class WordCount:
    def __init__(self, text_list, min_freq=0, max_freq=None, reserved_tokens=None):
        self.text_list = text_list
        self.min_freq = min_freq
        self.max_freq = max_freq  # None 表示不限制上限
        self.reserved_tokens = reserved_tokens or []
        self.word_count = dict()
        self.word_index = dict()
        self.count_word()
        self.build_word_index()
    
    def count_word(self):
        for text in self.text_list:
            for token in text:
                if token not in self.word_count:
                    self.word_count[token] = 1
                else:
                    self.word_count[token] += 1
        return self.word_count

    def build_word_index(self):
        self.index_word = dict()
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
            'min_freq': self.min_freq,
            'max_freq': self.max_freq,
            'reserved_tokens': self.reserved_tokens,
            'word_count': self.word_count,
            'word_index': self.word_index,
        }

    @classmethod
    def from_state(cls, state):
        vocab = cls(
            text_list=[],
            min_freq=state.get('min_freq', 0),
            max_freq=state.get('max_freq'),
            reserved_tokens=state.get('reserved_tokens', []),
        )
        vocab.word_count = dict(state.get('word_count', {}))
        vocab.word_index = dict(state['word_index'])
        vocab.index_word = {idx: token for token, idx in vocab.word_index.items()}
        return vocab


if __name__ =="__main__":
    print("hello world")
    text_list=["本书共1140个词条,分五个部分:第一部分1014条,包括有价证券及证券市场、股票、债券、票据、凭单",
    "本书才TI公司的DSP芯片为例，介绍了DSP的原理和应用方法，首先介绍了DSP的发展过程，引入TI公",
    "本书从交互设计的角度出发，讲述了用户研究的基础知识以及问卷、访谈、用户画像、数据分析等最常用的研究方",
    "The book is about about the user research and the user experience design, and the user research is a very important part of the user experience design."]
    
    token_list = tokenize_text(text_list, intro_tokenize_text)
    print(token_list)

    intro_word = WordCount(token_list)
    for i in range(10):
        print(intro_word[i])
        print(intro_word[intro_word[i]])


