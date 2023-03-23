import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class AeroscapesDataset(CustomDataset):

    CLASSES = ('background', 'person', 'bike', 'car', 'drone', 'boat', 'animal',
               'obstacle', 'construction', 'vegetation', 'road', 'sky')

    PALETTE = [[0, 0, 0], [192, 128, 128], [0, 128, 0], [128, 128, 128], [128, 0, 0], [0, 0, 128], [192, 0, 128],
               [192, 0, 0], [192, 128, 0], [0, 64, 0], [128, 128, 0], [0, 128, 128]]

    def __init__(self, split, **kwargs):
        super(AeroscapesDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
