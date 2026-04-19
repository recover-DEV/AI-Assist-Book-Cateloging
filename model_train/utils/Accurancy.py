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


def normalize_class_no(class_no):
    """Normalize a class number string for evaluation."""
    if class_no is None:
        return ""
    code = str(class_no).strip()
    if "/" in code:
        code = code.split("/")[0]
    return code


def hierarchical_accuracy(pred_list, label_list, levels=(1, 2, 3, 4)):
    """
    Compute hierarchical prefix accuracy for class numbers.

    Args:
        pred_list: iterable of predicted class numbers.
        label_list: iterable of gold class numbers.
        levels: prefix lengths used as hierarchy levels.

    Returns:
        dict:
            - total: number of evaluated pairs
            - exact_match: full-string accuracy
            - level@N: prefix accuracy at N characters
    """
    pred_len = len(pred_list)
    label_len = len(label_list)
    paired_len = min(pred_len, label_len)

    levels = sorted(set(int(l) for l in levels if int(l) > 0))
    total = 0
    exact_hits = 0
    level_hits = {level: 0 for level in levels}
    level_totals = {level: 0 for level in levels}

    for pred, label in zip(pred_list[:paired_len], label_list[:paired_len]):
        pred_code = normalize_class_no(pred)
        label_code = normalize_class_no(label)
        if not label_code:
            continue

        total += 1
        if pred_code == label_code:
            exact_hits += 1

        for level in levels:
            # Per-level denominator only counts samples whose gold label
            # has enough length for this hierarchy level.
            if len(label_code) < level:
                continue
            level_totals[level] += 1
            gold_prefix = label_code[:level]
            pred_prefix = pred_code[:level]
            if len(pred_code) >= level and pred_prefix == gold_prefix:
                level_hits[level] += 1

    if total == 0:
        result = {
            "total": 0,
            "exact_match": 0.0,
            "paired_samples": paired_len,
            "dropped_pred": max(0, pred_len - paired_len),
            "dropped_label": max(0, label_len - paired_len),
        }
        for level in levels:
            result[f"level@{level}"] = 0.0
            result[f"level@{level}_support"] = 0
        return result

    result = {
        "total": total,
        "exact_match": exact_hits / total,
        "paired_samples": paired_len,
        "dropped_pred": max(0, pred_len - paired_len),
        "dropped_label": max(0, label_len - paired_len),
    }
    for level in levels:
        support = level_totals[level]
        result[f"level@{level}"] = (level_hits[level] / support) if support > 0 else 0.0
        result[f"level@{level}_support"] = support
    return result