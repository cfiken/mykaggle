from typing import Optional, List
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

from mykaggle.lib.torch import label_smoothing


def get_loss_fn(
    loss_name: str,
    num_classes: int,
    *args, **kwargs
) -> nn.Module:
    if loss_name == 'ce':
        return nn.CrossEntropyLoss(
            reduction=kwargs['loss_reduction']
        )
    elif loss_name == 'custom_ce':
        return CustomCrossEntropy(
            num_classes=num_classes,
            reduction=kwargs['loss_reduction'],
            smoothing=kwargs['label_smoothing'],
        )
    elif loss_name == 'margin_rank':
        return nn.MarginRankingLoss(
            margin=kwargs['loss_rank_margin'], reduction=kwargs['loss_reduction']
        )
    elif loss_name == 'bce':
        return nn.BCEWithLogitsLoss(
            reduction=kwargs['loss_reduction']
        )
    elif loss_name == 'mse':
        return nn.MSELoss(
            reduction=kwargs['loss_reduction']
        )
    elif loss_name == 'mae':
        return nn.L1Loss(
            reduction=kwargs['loss_reduction']
        )
    elif loss_name == 'huber':
        return nn.HuberLoss(
            reduction=kwargs['loss_reduction'], delta=kwargs['loss_huber_delta']
        )
    elif loss_name == 'focal_cosine':
        return FocalCosineLoss(
            num_classes=num_classes,
            smoothing=kwargs['label_smoothing'],
            reduction=kwargs['loss_reduction'],
        )
    elif loss_name == 'gradually':
        loss_fns = []
        for _loss_name in kwargs['losses']:
            loss_fns.append(get_loss_fn(_loss_name, num_classes, *args, **kwargs))
        return GraduallyCombineLoss(
            loss_fns=loss_fns,
            transition_epoch=kwargs['loss_transition_epoch'],
            reduction=kwargs['loss_reduction']
        )
    return CustomCrossEntropy(
        num_classes=num_classes,
        reduction=kwargs['loss_reduction'],
        smoothing=kwargs['label_smoothing'],
    )


class CustomCrossEntropy(_WeightedLoss):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        num_classes: int = 1,
        smoothing: float = 0.0,
        device: str = 'cuda'
    ) -> None:
        super().__init__(weight=weight, reduction=reduction)
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        if len(targets.shape) < 2:  # do onehot
            targets = torch.eye(self.num_classes)[targets].to(targets.device)
        if self.smoothing != 0.0 and kwargs.get('do_smooth', True):
            targets = label_smoothing(targets, self.num_classes, self.smoothing)
        lsm = F.log_softmax(inputs, -1)
        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)
        loss = - (targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class FocalCosineLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        smoothing: float,
        alpha: float = 1.0,
        gamma: float = 2.0,
        xent: float = .1,
        reduction: str = 'mean'
    ):
        super(FocalCosineLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.alpha = alpha
        self.gamma = gamma

        self.xent = xent
        self.reduction = reduction

        self.y = torch.Tensor([1]).cuda()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs):
        if len(targets.shape) < 2:  # do onehot
            targets = torch.eye(self.num_classes)[targets].to(targets.device)
        if self.smoothing != 0.0 and kwargs.get('do_smooth', True):
            targets = label_smoothing(targets, self.num_classes, self.smoothing)
        cosine_loss = F.cosine_embedding_loss(
            inputs,
            targets,
            self.y,
            reduction=self.reduction
        )

        cent_loss = F.cross_entropy(F.normalize(inputs), torch.argmax(targets, -1), reduce=False)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss

        if self.reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss


class GraduallyCombineLoss(nn.Module):
    '''
    複数の loss を渡して、学習開始時は最初の loss を使用し、それ以降は指定した epoch まで
    残りの loss の平均を徐々に使うようにする。
    '''
    def __init__(
        self,
        loss_fns: List[nn.Module],  # [beginning, end1, end2, ...]
        transition_epoch: int,
        reduction: str = 'mean',
    ) -> None:
        '''
        Args:
          loss_fns: loss の nn.Module のインスタンス
          transition_epoch: 最初の loss を完全に使わなくなる epoch. この epoch まで線形に割合が変わる.
          reduction: reduction
        '''
        super().__init__()
        self.loss_fns = loss_fns
        self.transition_epoch = transition_epoch
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, epoch: int, **kwargs) -> torch.Tensor:
        loss_list = []
        for loss_fn in self.loss_fns:
            loss = loss_fn(inputs, targets, **kwargs)
            loss_list.append(loss)
        loss = torch.stack(loss_list)
        current = min(1.0, epoch / self.transition_epoch)
        output = (1 - current) * loss[0] + current * torch.mean(loss[1:], dim=0)
        if self.reduction == 'mean':
            output = torch.mean(output)
        return output
