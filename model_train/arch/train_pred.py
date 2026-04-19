import torch
import torch.nn as nn
import os
from utils.text_handle import WordCount, intro_tokenize_text, classify_tokenize_text, intro_tokenize_batch


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
        params = [p for p in net.parameters() if p.requires_grad and p.grad is not None]
    else:
        params = net.parameters()
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for param in params:
            param.grad.data *= theta / norm
    return norm


def train_seq2seq(net, data_iter, optimizer, num_epochs, tgt_vocab, device, save_dir, save_epoch, eval_epoch, start_epoch=0, val_iter=None, scheduler=None):
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
    best_val_loss = float('inf')
    best_model_path = None

    net.train()
    for epoch in range(num_epochs):
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing
            tgt_key_padding_mask = (
                torch.arange(dec_input.shape[1], device=device).unsqueeze(0)
                >= Y_valid_len.long().unsqueeze(1)
            )
            Y_hat, _ = net(X, dec_input, X_valid_len, tgt_key_padding_mask=tgt_key_padding_mask)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()

        if (epoch + 1) % save_epoch == 0:
            path = os.path.join(save_dir, "model" + str(epoch + start_epoch + 1) + ".pth")
            torch.save(net.state_dict(), path)
            print(f'model saved to {path}, train epoch {start_epoch + epoch + 1}')

        if (epoch + 1) % eval_epoch == 0:
            train_loss = l.sum() / num_tokens
            log_msg = f'epoch {epoch + 1}, train_loss {train_loss:.9f}'

            if val_iter is not None:
                val_loss = _eval_loss(net, val_iter, loss, tgt_vocab, device)
                log_msg += f', val_loss {val_loss:.9f}'

                if scheduler is not None:
                    scheduler.step(val_loss)
                    log_msg += f', lr {optimizer.param_groups[0]["lr"]:.2e}'

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(save_dir, "model_best.pth")
                    torch.save(net.state_dict(), best_model_path)
                    log_msg += '  ← best'

            print(log_msg)

    if best_model_path:
        print(f'最佳模型已保存至 {best_model_path}，验证 loss={best_val_loss:.9f}')


def _eval_loss(net, val_iter, loss_fn, tgt_vocab, device):
    net.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in val_iter:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            tgt_key_padding_mask = (
                torch.arange(dec_input.shape[1], device=device).unsqueeze(0)
                >= Y_valid_len.long().unsqueeze(1)
            )
            Y_hat, _ = net(X, dec_input, X_valid_len, tgt_key_padding_mask=tgt_key_padding_mask)
            l = loss_fn(Y_hat, Y, Y_valid_len)
            total_loss += l.sum().item()
            total_tokens += Y_valid_len.sum().item()
    net.train()
    return total_loss / total_tokens if total_tokens > 0 else float('inf')

def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False, title=None):
    net.eval()
    if title:
        title_tokens = intro_tokenize_text(title)
        intro_tokens = intro_tokenize_text(src_sentence)
        tokens = title_tokens + ['<sep>'] + intro_tokens
    else:
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


def predict_seq2seq_greedy(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False, title=None):
    net.eval()
    if title:
        title_tokens = intro_tokenize_text(title)
        intro_tokens = intro_tokenize_text(src_sentence)
        tokens = title_tokens + ['<sep>'] + intro_tokens
    else:
        tokens = intro_tokenize_text(src_sentence)
    unk = src_vocab['<unk>']
    src_tokens = [src_vocab[t] if t in src_vocab.word_index else unk for t in tokens] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq= []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        next_token = Y[:, -1:, :].argmax(dim=2)
        pred = next_token.squeeze().type(torch.int32).item()
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
        dec_X = torch.cat([dec_X, next_token], dim=1)
    return ''.join(tgt_vocab.index_word.get(i, '<unk>') for i in output_seq), 1


if __name__ == "__main__":
    from arch.encoder_decoder import EncoderDecoder
    text = "这本书是关于用户研究与用户体验设计的，用户研究是用户体验设计中非常重要的一部分。"
    label = "abcd1234567890ijdoqwnlkncuigwqbpmowoq,zlws"
    tokens = intro_tokenize_text(text)
    label_tokens = classify_tokenize_text(label)
    vocab = WordCount([tokens], min_freq=0, reserved_tokens=['<pad>', '<unk>', '<bos>', '<eos>'])
    label_vocab = WordCount([label_tokens], min_freq=0, reserved_tokens=['<pad>', '<unk>', '<bos>', '<eos>'])
    src_vocab_size = len(vocab)
    tgt_vocab_size = len(label_vocab)

    # # LSTM模型
    # from arch.LSTM import LSTM_encoder, LSTM_decoder
    # embedding_size = 256
    # num_hiddens = 256
    # num_layers = 2
    # dropout = 0.1
    # net = EncoderDecoder(LSTM_encoder(vocab_size, embedding_size, num_hiddens, num_layers, dropout), LSTM_decoder(vocab_size, embedding_size, num_hiddens, num_layers, dropout))
    # pred_seq, _ = predict_seq2seq(net, text, vocab, label_vocab, 30, device=torch.device('cpu'))


    # # Transformer模型
    # from arch.Transformer import TransformerEncoder, TransformerDecoder
    # d_model = 256
    # n_heads = 8
    # encoder_layers = 2
    # decoder_layers = 2
    # dim_feedforward = 512
    # dropout = 0.1
    # encoder = TransformerEncoder(src_vocab_size, d_model, n_heads, encoder_layers, dim_feedforward, dropout)
    # decoder = TransformerDecoder(tgt_vocab_size, d_model, n_heads, decoder_layers, dim_feedforward, dropout)
    # net = EncoderDecoder(encoder, decoder)

    # # encoder input: (batch_size, seq_len)(5,10)
    # # encoder output: (batch_size,seq_len, d_model)(5,10,256)
    # x = torch.randint(0, src_vocab_size, (5, 10), dtype=torch.long)
    # valid_len = torch.tensor([4, 5, 6, 7, 8], dtype=torch.long)
    # print(x.shape)
    # memory = net.encoder(x, valid_len)
    # dec_state = net.decoder.init_state(memory, valid_len)
    # # decoder input: (batch_size, seq_len)(5,7)
    # # decoder output: (batch_size, seq_len, tgt_vocab_size)(5,10,50)
    # label = torch.randint(0, tgt_vocab_size, (5, 7), dtype=torch.long)
    # valid_len = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    # y, dec_state = net.decoder(label, dec_state)
    # print(y.shape)
    # pred_seq,_ = predict_seq2seq_greedy(net, text, vocab, label_vocab, 30, device=torch.device('cpu'))
    # print(pred_seq)


    # BERT模型