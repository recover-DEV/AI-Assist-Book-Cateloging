import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import os
from utils.text_handle import WordCount, intro_tokenize_text, classify_tokenize_text, intro_tokenize_batch


def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__(label_smoothing=label_smoothing, reduction="none")

    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
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
        epoch_loss_sum = 0.0
        epoch_n_samples = 0
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
            epoch_loss_sum += float(l.sum().detach())
            epoch_n_samples += int(Y.size(0))
            l.sum().backward()
            grad_clipping(net, 1)
            optimizer.step()

        if (epoch + 1) % save_epoch == 0:
            path = os.path.join(save_dir, "model" + str(epoch + start_epoch + 1) + ".pth")
            torch.save(net.state_dict(), path)
            print(f'model saved to {path}, train epoch {start_epoch + epoch + 1}')

        if (epoch + 1) % eval_epoch == 0:
            train_loss = epoch_loss_sum / max(epoch_n_samples, 1)
            log_msg = f'epoch {epoch + 1}, train_loss {train_loss:.9f}'

            if val_iter is not None:
                val_loss = _eval_loss(net, val_iter, loss, tgt_vocab, device, is_bert_encoder=False)
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


def _eval_loss(net, val_iter, loss_fn, tgt_vocab, device, is_bert_encoder=False, use_amp=False):
    net.eval()
    total_loss, total_tokens = 0.0, 0
    amp_enabled = use_amp and device.type == 'cuda'
    with torch.no_grad():
        for batch in val_iter:
            if is_bert_encoder:
                input_ids, attention_mask, token_type_ids, X_valid_len, Y, Y_valid_len = [
                    x.to(device) for x in batch
                ]
                enc_X = {'input_ids': input_ids, 'attention_mask': attention_mask,
                         'token_type_ids': token_type_ids}
            else:
                X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
                enc_X = X
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            tgt_key_padding_mask = (
                torch.arange(dec_input.shape[1], device=device).unsqueeze(0)
                >= Y_valid_len.long().unsqueeze(1)
            )
            with autocast(device_type='cuda', enabled=amp_enabled):
                Y_hat, _ = net(enc_X, dec_input, X_valid_len, tgt_key_padding_mask=tgt_key_padding_mask)
            l = loss_fn(Y_hat.float(), Y, Y_valid_len)
            total_loss += float(l.sum())
            total_tokens += int(Y_valid_len.sum().item())
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

def train_bert_seq2seq(net, data_iter, optimizer, num_epochs, tgt_vocab, device, save_dir, save_epoch, eval_epoch, start_epoch=0, val_iter=None, scheduler=None, use_amp=None):
    if start_epoch == 0:
        def xavier_init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
            if type(m) == nn.GRU:
                for param in m._flat_weights_names:
                    if "weight" in param:
                        nn.init.xavier_uniform_(m._parameters[param])
        net.decoder.apply(xavier_init_weights)
    net.to(device)

    if use_amp is None:
        use_amp = device.type == 'cuda'
    amp_enabled = bool(use_amp) and device.type == 'cuda'
    scaler = GradScaler('cuda', enabled=amp_enabled)

    loss = MaskedSoftmaxCELoss()
    best_val_loss = float('inf')
    best_model_path = None

    net.train()
    for epoch in range(num_epochs):
        epoch_loss_sum = 0.0
        epoch_n_samples = 0
        for batch in data_iter:
            optimizer.zero_grad()
            input_ids, attention_mask, token_type_ids, X_valid_len, Y, Y_valid_len = [
                x.to(device) for x in batch
            ]
            enc_X = {'input_ids': input_ids, 'attention_mask': attention_mask,
                     'token_type_ids': token_type_ids}
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing
            tgt_key_padding_mask = (
                torch.arange(dec_input.shape[1], device=device).unsqueeze(0)
                >= Y_valid_len.long().unsqueeze(1)
            )
            with autocast(device_type='cuda', enabled=amp_enabled):
                Y_hat, _ = net(enc_X, dec_input, X_valid_len, tgt_key_padding_mask=tgt_key_padding_mask)
            l = loss(Y_hat.float(), Y, Y_valid_len)
            epoch_loss_sum += float(l.sum().detach())
            epoch_n_samples += int(Y.size(0))
            scaler.scale(l.sum()).backward()
            scaler.unscale_(optimizer)
            grad_clipping(net, 1)
            scaler.step(optimizer)
            scaler.update()

        if (epoch + 1) % save_epoch == 0:
            path = os.path.join(save_dir, "model" + str(epoch + start_epoch + 1) + ".pth")
            torch.save(net.state_dict(), path)
            print(f'model saved to {path}, train epoch {start_epoch + epoch + 1}')

        if (epoch + 1) % eval_epoch == 0:
            train_loss = epoch_loss_sum / max(epoch_n_samples, 1)
            log_msg = f'epoch {epoch + 1}, train_loss {train_loss:.9f}'

            if val_iter is not None:
                val_loss = _eval_loss(net, val_iter, loss, tgt_vocab, device, is_bert_encoder=True, use_amp=amp_enabled)
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



def predict_bert_seq2seq(net, title,intro,tgt_vocab, num_steps, device):
    net.eval()
    net.to(device)
    tokenize = net.encoder.tokenize_text(title, intro)
    tokenize = {k: tokenize[k].to(device) for k in tokenize.keys() if isinstance(tokenize[k], torch.Tensor)}
    feature = net.encoder(tokenize)['last_hidden_state']
    enc_valid_len = tokenize['attention_mask'].sum(dim=1).to(device)
    dec_state = net.decoder.init_state(feature, enc_valid_len)
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq = []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        next_token = Y[:, -1:, :].argmax(dim=2)
        pred = next_token.squeeze().type(torch.int32).item()
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
        dec_X = torch.cat([dec_X, next_token], dim=1)
    return ''.join(tgt_vocab.index_word.get(i, '<unk>') for i in output_seq), 1


def beam_search_bert_seq2seq(net, title, intro, tgt_vocab, num_steps, device, beam_width=3):
    """Beam search 解码。

    beam_width 条分支并行扩展；某分支预测到 <eos> 后停止并记录，其余分支继续，
    直到全部分支结束或达到 num_steps；返回长度归一化得分最高的分类号字符串。

    Args:
        net        : EncoderDecoder 模型
        title      : 书名字符串
        intro      : 简介字符串
        tgt_vocab  : WordCount 词表
        num_steps  : 最大生成步数
        device     : torch.device
        beam_width : 候选分支数（默认 3）

    Returns:
        (分类号字符串, beam_width)
    """
    import torch.nn.functional as F

    net.eval()
    net.to(device)

    bos_id = tgt_vocab['<bos>']
    eos_id = tgt_vocab['<eos>']

    with torch.no_grad():
        # ── 1. 编码输入 ──────────────────────────────────────────────────────
        tokenize = net.encoder.tokenize_text(title, intro)
        tokenize = {k: v.to(device) for k, v in tokenize.items() if isinstance(v, torch.Tensor)}
        memory        = net.encoder(tokenize)['last_hidden_state']  # [1, src_len, d_model]
        enc_valid_len = tokenize['attention_mask'].sum(dim=1)       # [1]

        # ── 2. 初始分支：1 条（第一步自动扩展到 beam_width）────────────────
        # active 中每个元素：(累积 log-prob, token_id 列表，含 bos)
        active    = [(0.0, [bos_id])]
        completed = []   # (score, token 列表，已去掉 bos/eos)

        for _ in range(num_steps):
            if not active:
                break

            n       = len(active)
            max_len = max(len(b[1]) for b in active)

            # ── 3. 组装解码器输入 [n, max_len] ──────────────────────────────
            seqs = torch.zeros(n, max_len, dtype=torch.long, device=device)
            for i, (_, toks) in enumerate(active):
                seqs[i, :len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)

            mem   = memory.expand(n, -1, -1)   # [n, src_len, d_model]
            enc_v = enc_valid_len.expand(n)    # [n]

            # ── 4. 解码器前向，取最后一步的 log 概率 ────────────────────────
            state     = net.decoder.init_state(mem, enc_v)
            Y, _      = net.decoder(seqs, state)                   # [n, seq_len, vocab]
            log_probs = F.log_softmax(Y[:, -1, :], dim=-1)         # [n, vocab]

            # ── 5. 每条活跃分支生成 beam_width 个候选，全局排序 ─────────────
            candidates = []
            for i, (score, toks) in enumerate(active):
                top_lp, top_ids = log_probs[i].topk(beam_width)
                for lp, tid in zip(top_lp.tolist(), top_ids.tolist()):
                    candidates.append((score + lp, toks + [tid]))
            candidates.sort(key=lambda x: x[0], reverse=True)

            # ── 6. 分流：eos → completed，其余保留为下一步活跃分支 ──────────
            new_active = []
            for score, toks in candidates:
                if toks[-1] == eos_id:
                    # 去掉 bos（第 0 位）和 eos（最后一位）
                    completed.append((score, toks[1:-1]))
                else:
                    new_active.append((score, toks))
                    if len(new_active) == beam_width:
                        break   # 已选够活跃分支，退出

            active = new_active

        # ── 7. 超出 num_steps 仍未结束的分支直接收尾（去掉 bos）────────────
        for score, toks in active:
            completed.append((score, toks[1:]))

        if not completed:
            return '', beam_width

        # ── 8. 长度归一化后取最优，避免短序列得分虚高 ──────────────────────
        def _length_norm(item):
            score, toks = item
            return score / max(len(toks), 1)

        best_score, best_toks = max(completed, key=_length_norm)
        return ''.join(tgt_vocab.index_word.get(t, '<unk>') for t in best_toks), beam_width

if __name__ == "__main__":
    from arch.encoder_decoder import EncoderDecoder
    title = ["书名"]
    text = ["这本书是关于用户研究与用户体验设计的，用户研究是用户体验设计中非常重要的一部分。"]
    label = "abcd1234567890ijdoqwnlkncuigwqbpmowoq,zlws"
    # tokens = intro_tokenize_text(text)
    label_tokens = classify_tokenize_text(label)
    # vocab = WordCount([tokens], min_freq=0, reserved_tokens=['<pad>', '<unk>', '<bos>', '<eos>'])
    label_vocab = WordCount([label_tokens], min_freq=0, reserved_tokens=['<pad>', '<unk>', '<bos>', '<eos>'])
    # src_vocab_size = len(vocab)
    tgt_vocab_size = len(label_vocab)

    # BERT模型
    from arch.BERT import BERT_encoder
    from arch.encoder_decoder import EncoderDecoder
    from arch.Transformer import TransformerDecoder
    encoder = BERT_encoder(max_length=128)
    d_model = 768
    n_heads = 8
    decoder_layers = 2
    dim_feedforward = 512
    dropout = 0.3
    decoder = TransformerDecoder(tgt_vocab_size, d_model, n_heads, decoder_layers, dim_feedforward, dropout)
    net = EncoderDecoder(encoder, decoder)

    pred_seq,_ = predict_bert_seq2seq(net, title, text, label_vocab, 30, device=torch.device('cpu'))
    print(pred_seq)

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


