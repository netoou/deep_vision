import torch

def topk_accuracy(logits: torch.tensor, targets: torch.tensor, k: int):
    _, topk = logits.softmax(dim=1).topk(k=k, dim=1)
    true_cnt = 0
    for pred, tgt in zip(topk, targets):
        if tgt in pred:
            true_cnt += 1

    return true_cnt / len(targets)