# Copyright (c) OpenMMLab. All rights reserved.

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class M4DSegDataset(CustomDataset):
    """M4D multi-class SAR oil spill dataset (paper §4.1).

    Five semantic classes:
      0 sea_surface, 1 oil_spill, 2 look_alike, 3 vessel, 4 land
    """

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
