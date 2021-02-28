import torch


def label_smoothing(targets: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    '''
    onehot な target を受け取って smoothing して返す。
    Args:
      targets: target
      num_classes: クラス数
      smoothing: label smoothing の値
    '''
    assert 0 <= smoothing < 1
    with torch.no_grad():
        targets = targets - targets * smoothing
        targets = targets + smoothing / num_classes
    return targets
