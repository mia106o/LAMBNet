# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def extract_boundary_sobel(mask, kernel_size=3):
    """Extract one-pixel-wide contours via Sobel (L1 gradient magnitude).

    Manuscript: boundary masks are obtained with the Sobel operator and have a
    fixed width of one pixel (no morphological dilation).
    """
    sobel_x = mask.new_tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = mask.new_tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    gx = F.conv2d(mask, sobel_x, padding=1)
    gy = F.conv2d(mask, sobel_y, padding=1)
    # L1-norm gradient magnitude (manuscript BEM / Boundary ablation setup)
    mag = torch.abs(gx) + torch.abs(gy)
    return (mag > 0).float()


def extract_oil_boundary(target, oil_class=1, kernel_size=3):
    """Extract oil-spill contour pixels from semantic labels (binary SOS)."""
    oil_mask = (target == oil_class).float().unsqueeze(1)
    return extract_boundary_sobel(oil_mask, kernel_size=kernel_size)


@LOSSES.register_module()
class BoundaryLoss(nn.Module):
    """Boundary-aware loss on ground-truth contour pixels only.

    Manuscript Eq.(9): loss is computed only over GT boundary pixels extracted
    by Sobel; if a sample has no boundary pixels, the term is set to zero.
    """

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

    def _binary_boundary_loss(self, pred_logit, gt_mask, valid_mask):
        """pred_logit: (N,H,W) logits or probs; gt_mask: (N,H,W) in {0,1}."""
        boundary = extract_boundary_sobel(gt_mask.unsqueeze(1)).squeeze(1)
        boundary = boundary * valid_mask
        denom = boundary.sum()
        if denom <= 0:
            return pred_logit.new_zeros(())
        pixel_loss = F.binary_cross_entropy_with_logits(
            pred_logit, gt_mask, reduction='none')
        return (pixel_loss * boundary).sum() / denom

    def forward(self, pred, target, weight=None, ignore_index=255, **kwargs):
        if ignore_index is None:
            ignore_index = self.ignore_index

        target = target.squeeze(1) if target.dim() == 4 else target
        valid_mask = (target != ignore_index).float()

        if self.multi_class:
            # Manuscript Eq.(15): per-class one-hot + Sobel boundaries
            num_classes = pred.shape[1]
            total = pred.new_zeros(())
            active = 0
            for c in range(num_classes):
                gt_c = (target == c).float()
                if gt_c.sum() <= 0:
                    continue
                loss_c = self._binary_boundary_loss(pred[:, c], gt_c, valid_mask)
                total = total + loss_c
                active += 1
            if active == 0:
                return pred.new_zeros(())
            return self.loss_weight * (total / active)

        # Binary SOS: oil class contour only (Eq.9)
        pred_oil = pred[:, self.oil_class] if pred.shape[1] > 1 else pred[:, 0]
        gt_oil = (target == self.oil_class).float()
        loss = self._binary_boundary_loss(pred_oil, gt_oil, valid_mask)
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self._loss_name
