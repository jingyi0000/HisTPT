# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

from modeling.utils import configurable
from torchvision import transforms
from PIL import Image

__all__ = ["MaskFormerSemanticDatasetMapper"]

class MaskFormerSemanticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        min_size_test, #add for test-time eval
        max_size_test, #add for test-time eval
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.min_size_test = min_size_test #add for test-time eval
        self.max_size_test = max_size_test #add for test-time eval

        t = []
        t.append(transforms.Resize(self.min_size_test, interpolation=Image.BICUBIC))
        self.transform = transforms.Compose(t)

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        cfg_input = cfg['INPUT']
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg_input['MIN_SIZE_TRAIN'],
                cfg_input['MAX_SIZE_TRAIN'],
                cfg_input['MIN_SIZE_TRAIN_SAMPLING'],
            )
        ]
        cfg_input_crop = cfg_input['CROP']
        if cfg_input_crop['ENABLED']:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg_input_crop['TYPE'],
                    cfg_input_crop['SIZE'],
                    cfg_input_crop['SINGLE_CATEGORY_MAX_AREA'],
                    cfg_input['IGNORE_VALUE'],
                )
            )
        # if cfg_input['COLOR_AUG_SSD']:
        #     augs.append(ColorAugSSDTransform(img_format=cfg_input['FORMAT']))
        # augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg['DATASETS']['TRAIN']
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg_input['FORMAT'],
            "ignore_label": ignore_label,
            "size_divisibility": cfg_input['SIZE_DIVISIBILITY'],
            "min_size_test": cfg_input['MIN_SIZE_TEST'], #add for test-time eval
            "max_size_test": cfg_input['MAX_SIZE_TEST'], #add for test-time eval
        }
        return ret
    
    def read_semseg(self, file_name): #add for test-time eval
        if '.png' in file_name:
            semseg = np.asarray(Image.open(file_name))
        elif '.mat' in file_name:
            semseg = scipy.io.loadmat(file_name)['LabelMap']
        return semseg

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        image_ori = image #add for augmentations
        utils.check_image_size(dataset_dict, image)

        #add for test-time eval seperately
        file_name_eval = dataset_dict['file_name']
        semseg_name_eval = dataset_dict['sem_seg_file_name']

        if "sem_seg_file_name" in dataset_dict:      
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double") #original one
            sem_seg_gt_ori = sem_seg_gt #add for augmentations
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        #for test time augmentations  [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        #1
        aug_input_1 = T.AugInput(image_ori, sem_seg=sem_seg_gt_ori)
        augmentation_1 = [T.ResizeShortestEdge(
                768,
                2048,
                'choice',
            )
            ]
        aug_input_1, transforms_1 = T.apply_transform_gens(augmentation_1, aug_input_1)
        #2
        aug_input_2 = T.AugInput(image_ori, sem_seg=sem_seg_gt_ori)
        augmentation_2 = [T.ResizeShortestEdge(
                1024,
                2048,
                'choice',
            )
            ]
        aug_input_2, transforms_2 = T.apply_transform_gens(augmentation_2, aug_input_2)
        # 
        image_aug_1 = aug_input_1.image
        image_aug_2 = aug_input_2.image
        image_aug_1 = torch.as_tensor(np.ascontiguousarray(image_aug_1.transpose(2, 0, 1)))
        image_aug_2 = torch.as_tensor(np.ascontiguousarray(image_aug_2.transpose(2, 0, 1)))

        dataset_dict["image_aug"] = image_aug_1
        dataset_dict["image_ori"] = image_aug_2
        

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        #add for test-time eval seperately
        image_eval = Image.open(file_name_eval).convert('RGB')

        dataset_dict['width'] = image_eval.size[0]
        dataset_dict['height'] = image_eval.size[1]

        image_eval = self.transform(image_eval)
        image_eval = torch.from_numpy(np.asarray(image_eval).copy())
        image_eval = image_eval.permute(2,0,1)
            
        semseg_eval = self.read_semseg(semseg_name_eval)
        semseg_eval = torch.from_numpy(semseg_eval.astype(np.int32))
        dataset_dict['image_eval'] = image_eval
        dataset_dict['semseg_eval'] = semseg_eval

        return dataset_dict
