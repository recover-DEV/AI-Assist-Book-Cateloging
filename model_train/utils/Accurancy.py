import math
import collections

def bleu(pred_seq, label_seq, k):
    pred_tokens, label_tokens = list(pred_seq), list(label_seq)
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    
    if len_pred == 0:          # 预测为空，BLEU 直接为 0
        return 0.0
    
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        denom = len_pred - n + 1
        if denom <= 0:
            # 序列过短无法计算 n-gram，改为统计各 token 在实际序列中的匹配比例
            label_counter = collections.Counter(label_tokens)
            pred_counter = collections.Counter(pred_tokens)
            char_matches = sum(min(cnt, label_counter[tok]) for tok, cnt in pred_counter.items())
            return char_matches / len_label
        for i in range(denom):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / denom, math.pow(0.5, n))
    return score