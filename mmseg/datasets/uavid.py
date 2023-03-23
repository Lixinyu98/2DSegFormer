import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class UavidDataset(CustomDataset):

    CLASSES = ('background clutter', 'Building', 'Road', 'Static car', 'Tree', 'Low vegetation', 'Human', 'Moving car')

    PALETTE = [[0, 0, 0], [128, 0, 0], [128, 64, 128], [192, 0, 192], [0, 128, 0], [128, 128, 0], [64, 64, 0], [64, 0, 128]]

    def __init__(self, **kwargs):
        super(UavidDataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', **kwargs)