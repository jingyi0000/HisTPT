# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import numpy as np
import os
import glob
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from utils.constants import CITYSCAPES


from detectron2.data.datasets.cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from detectron2.data.datasets.cityscapes_panoptic import register_all_cityscapes_panoptic

__all__ = ["register_cityscapes_instance"]



_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cityscapes_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
    "cityscapes_{task}_test": ("cityscapes/leftImg8bit/test/", "cityscapes/gtFine/test/"),
}


def register_cityscapes_instance(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_ins_box", **meta
        )





# _root = os.getenv("DATASET", "datasets")
_root = 'xdecoder_data'
register_cityscapes_instance(_root)