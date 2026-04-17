from arch.encoder_decoder import Encoder, Decoder, EncoderDecoder
from utils.text_handle import intro_tokenize_text, WordCount,classify_tokenize_text
from torch.nn import LSTM
import torch
import os
from torch import nn

class LSTM_encoder(Encoder):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers,dropout=0.0):
        super(LSTM_encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
    
    def forward(self,X, *args):
        X = self.embedding(X).permute(1,0,2)
        output, state = self.rnn(X)
        return output, state

class LSTM_decoder(Decoder):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers,dropout=0.0):
        super(LSTM_decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = LSTM(embedding_size+hidden_size, hidden_size, num_layers, dropout=dropout)
        self.dense = nn.Linear(hidden_size, vocab_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def init_state(self,enc_outputs, *args):
        return enc_outputs[1]
    
    def forward(self,X,state):
        X = self.embedding(X).permute(1,0,2)
        context = state[0][-1].repeat(X.shape[0],1,1)
        X = torch.cat((X,context), 2)
        output, state = self.rnn(X,state)
        output = self.layer_norm(output)
        output = self.dense(output).permute(1,0,2)
        return output, state

def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.parameters()
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for param in params:
            param.grad.data *= theta / norm
    return norm


def train_seq2seq(net, data_iter, optimizer, num_epochs, tgt_vocab, device, save_dir, save_epoch, eval_epoch, start_epoch=0):
    if start_epoch == 0:
        def xavier_init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
            if type(m) == nn.GRU:
                for param in m._flat_weights_names:
                    if "weight" in param:
                        nn.init.xavier_uniform_(m._parameters[param])
        net.apply(xavier_init_weights)
    net.to(device)
    
    loss = MaskedSoftmaxCELoss()
    net.train()
    for epoch in range(num_epochs):
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
        if (epoch + 1) % save_epoch == 0:
            path = os.path.join(save_dir, "model"+str(epoch+start_epoch + 1)+".pth")
            torch.save(net.state_dict(), path)
            print(f'model saved to {path}, train epoch {start_epoch + epoch + 1}')
        if (epoch + 1) % eval_epoch == 0:
            print(f'epoch {epoch + 1}, loss {l.sum() / num_tokens:.6f}')

def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    # Set `net` to eval mode for inference
    net.eval()
    tokens = intro_tokenize_text(src_sentence)
    unk = src_vocab['<unk>']
    src_tokens = [src_vocab[t] if t in src_vocab.word_index else unk for t in tokens] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ''.join(tgt_vocab.index_word.get(i, '<unk>') for i in output_seq), attention_weight_seq


if __name__ == "__main__":
    text = "这本书是关于用户研究与用户体验设计的，用户研究是用户体验设计中非常重要的一部分。"
    label = "abcd1234567890ijdoqwnlkncuigwqbpmowoq,zlws"
    vocab_size = 1000
    embedding_size = 256
    num_hiddens = 256
    num_layers = 2
    dropout = 0.1
    net = EncoderDecoder(LSTM_encoder(vocab_size, embedding_size, num_hiddens, num_layers, dropout), LSTM_decoder(vocab_size, embedding_size, num_hiddens, num_layers, dropout))
    tokens = intro_tokenize_text(text)
    label_tokens = classify_tokenize_text(label)
    vocab = WordCount([tokens], min_freq=0, reserved_tokens=['<pad>', '<unk>', '<bos>', '<eos>'])
    label_vocab = WordCount([label_tokens], min_freq=0, reserved_tokens=['<pad>', '<unk>', '<bos>', '<eos>'])
    pred_seq, label_seq = predict_seq2seq(net, text, vocab, label_vocab, 30, device=torch.device('cpu'))
    print(pred_seq)
    print(label_seq)