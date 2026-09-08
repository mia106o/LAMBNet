# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def _sobel_l1_magnitude(feat):
    """Sobel L1-norm gradient magnitude (manuscript GP / BEM setting)."""
    sobel_x = feat.new_tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = feat.new_tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    gx = F.conv2d(feat, sobel_x, padding=1)
    gy = F.conv2d(feat, sobel_y, padding=1)
    return torch.abs(gx) + torch.abs(gy)


@LOSSES.register_module()
class GradientPenaltyLoss(nn.Module):
    """Gradient-profile (GP) loss: Sobel gradient consistency on oil edges.

    Manuscript Eq.(10)/(16): horizontal & vertical Sobel gradients between
    prediction and ground-truth probability maps.
    """

    def __init__(self,
                 oil_class=1,
                 loss_weight=1.0,
                 ignore_index=255,
                 multi_class=False,
                 loss_name='loss_gp'):
        super(GradientPenaltyLoss, self).__init__()
        self.oil_class = oil_class
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.multi_class = multi_class
        self._loss_name = loss_name

    def forward(self, pred, target, weight=None, ignore_index=255, **kwargs):
        if ignore_index is None:
            ignore_index = self.ignore_index

        target = target.squeeze(1) if target.dim() == 4 else target
        valid_mask = (target != ignore_index).float().unsqueeze(1)

        if self.multi_class:
            # Softmax probs; sum GP over classes (Eq.16)
            probs = F.softmax(pred, dim=1)
            num_classes = probs.shape[1]
            total = pred.new_zeros(())
            for c in range(num_classes):
                gt_c = (target == c).float().unsqueeze(1)
                pred_grad = _sobel_l1_magnitude(probs[:, c:c + 1])
                gt_grad = _sobel_l1_magnitude(gt_c)
                diff = torch.abs(pred_grad - gt_grad) * valid_mask
                denom = valid_mask.sum().clamp(min=1.0)
                total = total + diff.sum() / denom
            return self.loss_weight * (total / num_classes)

        # Binary SOS: oil probability channel (Eq.10)
        if pred.shape[1] == 1:
            pred_oil = torch.sigmoid(pred)
        else:
            pred_oil = F.softmax(pred, dim=1)[:, self.oil_class:self.oil_class + 1]
        gt_oil = (target == self.oil_class).float().unsqueeze(1)

        pred_grad = _sobel_l1_magnitude(pred_oil)
        gt_grad = _sobel_l1_magnitude(gt_oil)

        diff = torch.abs(pred_grad - gt_grad) * valid_mask
        denom = valid_mask.sum().clamp(min=1.0)
        loss = diff.sum() / denom
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self._loss_name
