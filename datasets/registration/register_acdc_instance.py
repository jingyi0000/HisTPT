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


__all__ = ["register_acdc_instance"]


def register_acdc_instance(root):
    SPLITS = [
            ("fog_acdc_val_ins_seg", "ACDC/rgb_anon", "ACDC/gt_detection_trainval/gt_detection/fog/instancesonly_fog_val_gt_detection.json", "fog"),
            ("night_acdc_val_ins_seg", "ACDC/rgb_anon", "ACDC/gt_detection_trainval/gt_detection/night/instancesonly_night_val_gt_detection.json", "night"),
            ("rain_acdc_val_ins_seg", "ACDC/rgb_anon", "ACDC/gt_detection_trainval/gt_detection/rain/instancesonly_rain_val_gt_detection.json", "rain"),
            ("snow_acdc_val_ins_seg", "ACDC/rgb_anon", "ACDC/gt_detection_trainval/gt_detection/snow/instancesonly_snow_val_gt_detection.json", "snow"),
        ]
        
    for name, dirname, gtdir, split in SPLITS:
        register_coco_instances(name, {}, os.path.join(root, gtdir), os.path.join(root, dirname))
        # MetadataCatalog.get(name).evaluator_type = "sem_seg"


# _root = os.getenv("DATASET", "datasets")
_root = 'xdecoder_data'
register_acdc_instance(_root)