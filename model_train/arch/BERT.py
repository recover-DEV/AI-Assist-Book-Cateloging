from arch.encoder_decoder import Encoder, Decoder, EncoderDecoder
from transformers import BertModel, BertTokenizer
import torch

class BERT_encoder(Encoder):
    def __init__(self, model_path='bert-base-chinese', max_length=256, n_freeze_layers=4, freeze_bert=False):
        super(BERT_encoder, self).__init__()
        self.max_length = max_length
        self.bert = BertModel.from_pretrained(model_path)
        # 只冻结前 n_freeze_layers 个 Transformer 层，Embedding 默认不冻结
        for i in range(n_freeze_layers):
            for p in self.bert.encoder.layer[i].parameters():
                p.requires_grad = False
        # freeze_bert 保留向后兼容：为 True 时额外冻结 Embedding
        if freeze_bert:
            for p in self.bert.embeddings.parameters():
                p.requires_grad = False
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_path)
    
    def tokenize_text(self, title_list, intro_list):
        if isinstance(title_list, str):
            title_list = [title_list]
        if isinstance(intro_list, str):
            intro_list = [intro_list]
        return self.bert_tokenizer(
            title_list,
            intro_list,
            padding=True,
            truncation='only_second',
            max_length=self.max_length,
            return_tensors='pt',
        )

    def forward(self, x, *args, **kwargs):
        if not isinstance(x, dict):
            raise TypeError('BERT_encoder expects tokenizer output dict with input_ids, attention_mask, …')
        return self.bert(**x)
    
    

if __name__ == '__main__':

    # batch_size = 3, seq_len = 16, hidden_size = 768
    title_list = ["书名1", "书名2", "书名3"]
    intro_list = ["这本书是", "这本书", "这"]
    encoder = BERT_encoder(max_length=16)
    encoding = encoder.tokenize_text(title_list, intro_list)
    print(encoding)
    outputs = encoder(encoding)
    print(outputs['last_hidden_state'].shape)
    print(outputs['pooler_output'].shape)
    label = "TN12"