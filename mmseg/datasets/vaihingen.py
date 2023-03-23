from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class VaigingenDataset(CustomDataset):
    CLASSES = (
        'Impervious surfaces', 'Buildings', 'Low vegetation', 'Trees', 'Cars')

    PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0]]

    def __init__(self, **kwargs):
        super(VaigingenDataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='.tif',
            reduce_zero_label=True,
            **kwargs)
