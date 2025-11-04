def f1_exact(pred_lines, gold_lines):
    pred, gold = set(pred_lines), set(gold_lines)
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return prec, rec, f1