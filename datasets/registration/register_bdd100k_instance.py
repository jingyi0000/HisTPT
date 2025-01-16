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

from detectron2.data.datasets import register_coco_instances


from utils.constants import CITYSCAPES


__all__ = ["register_bdd100k_instance"]


def register_bdd100k_instance(root):
    SPLITS = [
            ("bdd10k_val_ins_seg", "BDD100k/bdd100k/images/10k/val", "BDD100k/bdd100k/labels/ins_seg_val_coco.json", "val")
        ]
        
    for name, dirname, gtdir, split in SPLITS:
        register_coco_instances(name, {}, os.path.join(root, gtdir), os.path.join(root, dirname))
        # MetadataCatalog.get(name).evaluator_type = "sem_seg"


# _root = os.getenv("DATASET", "datasets")
_root = 'xdecoder_data'
register_bdd100k_instance(_root)