from microtorch_py.engine import Tensor


def binary_cross_entropy(logits, target):
    probs = logits.sigmoid()
    safe_probs = probs + 1e-8
    safe_ones = 1.0 - probs + 1e-8
    loss = -(target * safe_probs.log() + (1 - target) * safe_ones.log())
    return loss.sum()
