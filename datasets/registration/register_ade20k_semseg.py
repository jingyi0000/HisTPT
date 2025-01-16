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

from utils.constants import ADE_PANOPTIC_CLASSES

__all__ = ["load_ade_instances", "register_ade_context"]


def load_ade_instances(name: str, dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load ADE annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    img_folder = os.path.join(dirname, 'images', split)
    img_pths = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))
    
    sem_folder = os.path.join(dirname, 'annotations_detectron2', split)
    sem_pths = sorted(glob.glob(os.path.join(sem_folder, '*.png')))

    assert len(img_pths) == len(sem_pths)
        
    dicts = []
    for img_pth, sem_pth in zip(img_pths, sem_pths):
        r = {
            "file_name": img_pth,
            "sem_seg_file_name": sem_pth,
            "image_id": img_pth.split('/')[-1].split('.')[0],
        }
        dicts.append(r)
    return dicts


def register_ade_context(name, dirname, split, class_names=ADE_PANOPTIC_CLASSES):
    DatasetCatalog.register(name, lambda: load_ade_instances(name, dirname, split, class_names))
    MetadataCatalog.get(name).set(
        stuff_classes=class_names,
        dirname=dirname,
        split=split,
        ignore_label=[255],
        thing_dataset_id_to_contiguous_id={},
        class_offset=0,
        keep_sem_bgd=False
    )


def register_ade_semseg(root):
    SPLITS = [
            ("ade20k_val_sem_seg", "ADE20K/ADEChallengeData2016", "val"),
        ]
        
    for name, dirname, split in SPLITS:
        register_ade_context(name, os.path.join(root, dirname), split)
        MetadataCatalog.get(name).evaluator_type = "sem_seg"


_root = 'xdecoder_data'
register_ade_semseg(_root)