# Copyright (c) OpenMMLab. All rights reserved.

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class M4DSegDataset(CustomDataset):

    CLASSES = ('sea_surface', 'oil_spill', 'look_alike', 'vessel', 'land')
    PALETTE = [
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 0],
        [255, 255, 0],
        [128, 128, 128],
    ]

    def __init__(self, **kwargs):
        super().__init__(
            reduce_zero_label=False,
            **kwargs)
