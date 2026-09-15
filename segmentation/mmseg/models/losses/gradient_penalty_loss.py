
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def _sobel_xy(feat):
    sobel_x = feat.new_tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = feat.new_tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    gx = F.conv2d(feat, sobel_x, padding=1)
    gy = F.conv2d(feat, sobel_y, padding=1)
    return gx, gy


@LOSSES.register_module()
class GradientPenaltyLoss(nn.Module):

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

    def _gp_term(self, pred_map, gt_map, valid_mask):
        gx_p, gy_p = _sobel_xy(pred_map)
        gx_g, gy_g = _sobel_xy(gt_map)
        diff = (gx_p - gx_g) ** 2 + (gy_p - gy_g) ** 2
        diff = diff * valid_mask
        n = valid_mask.sum().clamp(min=1.0)
        return diff.sum() / (2.0 * n)

    def forward(self, pred, target, weight=None, ignore_index=255, **kwargs):
        if ignore_index is None:
            ignore_index = self.ignore_index

        target = target.squeeze(1) if target.dim() == 4 else target
        valid_mask = (target != ignore_index).float().unsqueeze(1)

        if self.multi_class:
            probs = F.softmax(pred, dim=1)
            num_classes = probs.shape[1]
            total = pred.new_zeros(())
            for c in range(num_classes):
                gt_c = (target == c).float().unsqueeze(1)
                total = total + self._gp_term(
                    probs[:, c:c + 1], gt_c, valid_mask)
            return self.loss_weight * (total / num_classes)

        if pred.shape[1] == 1:
            pred_oil = torch.sigmoid(pred)
        else:
            pred_oil = torch.sigmoid(pred[:, self.oil_class:self.oil_class + 1])
        gt_oil = (target == self.oil_class).float().unsqueeze(1)
        loss = self._gp_term(pred_oil, gt_oil, valid_mask)
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self._loss_name
