
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def extract_boundary_sobel(mask, kernel_size=3):
    sobel_x = mask.new_tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = mask.new_tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    gx = F.conv2d(mask, sobel_x, padding=1)
    gy = F.conv2d(mask, sobel_y, padding=1)
    mag = torch.abs(gx) + torch.abs(gy)
    return (mag > 0).float()


def extract_oil_boundary(target, oil_class=1, kernel_size=3):
    oil_mask = (target == oil_class).float().unsqueeze(1)
    return extract_boundary_sobel(oil_mask, kernel_size=kernel_size)


@LOSSES.register_module()
class BoundaryLoss(nn.Module):

    def __init__(self,
                 oil_class=1,
                 loss_weight=1.0,
                 ignore_index=255,
                 multi_class=False,
                 loss_name='loss_boundary'):
        super(BoundaryLoss, self).__init__()
        self.oil_class = oil_class
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.multi_class = multi_class
        self._loss_name = loss_name

    def _binary_boundary_loss(self, pred_prob, gt_mask, valid_mask):
        boundary = extract_boundary_sobel(gt_mask.unsqueeze(1)).squeeze(1)
        boundary = boundary * valid_mask
        denom = boundary.sum()
        if denom <= 0:
            return pred_prob.new_zeros(())
        pixel_loss = F.binary_cross_entropy(pred_prob, gt_mask, reduction='none')
        return (pixel_loss * boundary).sum() / denom

    def forward(self, pred, target, weight=None, ignore_index=255, **kwargs):
        if ignore_index is None:
            ignore_index = self.ignore_index

        target = target.squeeze(1) if target.dim() == 4 else target
        valid_mask = (target != ignore_index).float()

        if self.multi_class:
            num_classes = pred.shape[1]
            probs = F.softmax(pred, dim=1)
            total = pred.new_zeros(())
            for c in range(num_classes):
                gt_c = (target == c).float()
                loss_c = self._binary_boundary_loss(probs[:, c], gt_c, valid_mask)
                total = total + loss_c
            return self.loss_weight * (total / num_classes)

        if pred.shape[1] == 1:
            pred_oil = torch.sigmoid(pred[:, 0])
            gt_oil = (target == self.oil_class).float()
        else:
            pred_oil = torch.sigmoid(pred[:, self.oil_class])
            gt_oil = (target == self.oil_class).float()
        loss = self._binary_boundary_loss(pred_oil, gt_oil, valid_mask)
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self._loss_name
