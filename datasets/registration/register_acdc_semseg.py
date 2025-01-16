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

from utils.constants import CITYSCAPES

__all__ = ["load_acdc_semantic", "register_acdc_context"]



def _get_acdc_files(image_dir, gt_dir):
    files = []
    # scan through the directory
    cities = PathManager.ls(image_dir)
    # logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        for basename in PathManager.ls(city_img_dir):
            image_file = os.path.join(city_img_dir, basename)

            suffix = "rgb_anon.png"
            assert basename.endswith(suffix), basename
            basename = basename[: -len(suffix)]

            instance_file = os.path.join(city_gt_dir, basename + "gt_invIds.png")
            label_file = os.path.join(city_gt_dir, basename + "gt_labelIds.png")
            # json_file = os.path.join(city_gt_dir, basename + "gtFine_polygons.json")

            files.append((image_file, instance_file, label_file))
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def load_acdc_semantic(name, dirname, gtdir, split, class_names):

    dirname_split = os.path.join(dirname, split, 'val')
    gtdir_split = os.path.join(gtdir, split, 'val')

    ret = []
    gt_dir = PathManager.get_local_path(gtdir_split)
    image_dir = dirname_split

    for image_file, _, label_file in _get_acdc_files(image_dir, gt_dir):
        label_file = label_file.replace("labelIds", "labelTrainIds")

        ret.append(
            {
                "file_name": image_file,
                "sem_seg_file_name": label_file,
                "height": int(1080),
                "width": int(1920),
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
    return ret




def register_acdc_context(name, dirname, gtdir, split, class_names=CITYSCAPES):
    DatasetCatalog.register(name, lambda: load_acdc_semantic(name, dirname, gtdir, split, class_names))
    MetadataCatalog.get(name).set(
        stuff_classes=class_names,
        dirname=dirname,
        split=split,
        ignore_label=[255],
        thing_dataset_id_to_contiguous_id={},
        class_offset=0,
        keep_sem_bgd=False
    )




def register_acdc_semseg(root):
    SPLITS = [
            ("fog_acdc_val_sem_seg", "ACDC/rgb_anon", "ACDC/gt", "fog"),
            ("night_acdc_val_sem_seg", "ACDC/rgb_anon", "ACDC/gt", "night"),
            ("rain_acdc_val_sem_seg", "ACDC/rgb_anon", "ACDC/gt", "rain"),
            ("snow_acdc_val_sem_seg", "ACDC/rgb_anon", "ACDC/gt", "snow"),
        ]
        
    for name, dirname, gtdir, split in SPLITS:
        register_acdc_context(name, os.path.join(root, dirname), os.path.join(root, gtdir),split)
        MetadataCatalog.get(name).evaluator_type = "sem_seg"


# _root = os.getenv("DATASET", "datasets")
_root = 'xdecoder_data'
register_acdc_semseg(_root)