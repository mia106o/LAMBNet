# Copyright (c) OpenMMLab. All rights reserved.

from .builder import DATASETS
from .custom import CustomDataset
from .pipelines.loading import format_sos_seg_map


@DATASETS.register_module()
class SOSSegDataset(CustomDataset):
    """SOS (CBD-Net) oil spill segmentation dataset."""

    CLASSES = ('background', 'oil_spill')
    PALETTE = [[0, 0, 0], [255, 0, 0]]

    def __init__(self,
                 img_suffix='_sat.jpg',
                 seg_map_suffix='_mask.png',
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)

    def get_gt_seg_map_by_idx(self, index):
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        return format_sos_seg_map(results['gt_semantic_seg'])

    def get_gt_seg_maps(self, efficient_test=None):
        for idx in range(len(self)):
            yield self.get_gt_seg_map_by_idx(idx)
