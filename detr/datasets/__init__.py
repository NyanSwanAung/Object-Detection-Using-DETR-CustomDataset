# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision
from .coco import build as build_coco


## ********* For face dataset *************** ##
from .face import build as build_face
from .covid import build as build_covid
## ******************************************##


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)

    ## ****************************** ##
    if args.dataset_file == 'face':
        return build_face(image_set, args)

    if args.dataset_file == 'siim-covid':
        return build_covid(image_set, args)
    ## ****************************** ##

    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')
