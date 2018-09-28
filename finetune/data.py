import numpy as np
import collections
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision import transforms
from albumentations import (Resize, JpegCompression, Normalize,
    HorizontalFlip, ShiftScaleRotate, CLAHE, Blur, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, OneOf, Compose)

from prometheus.utils.parse import parse_in_csvs
from prometheus.utils.factory import UtilsFactory
from prometheus.data.reader import ImageReader, ScalarReader, ReaderCompose
from prometheus.data.augmentor import Augmentor
from prometheus.data.sampler import BalanceClassSampler
from prometheus.dl.datasource import AbstractDataSource

# ---- Augmentations ----


IMG_SIZE = 224


def strong_aug(p=.5):
    return Compose([
        JpegCompression(p=0.9),
        HorizontalFlip(p=0.5),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.5),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=.1),
            Blur(blur_limit=3, p=.1),
        ], p=0.5),
        ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=.5),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.5),
        HueSaturationValue(p=0.5),
    ], p=p)


AUG_TRAIN = strong_aug(p=0.75)
AUG_INFER = Compose([
    Resize(IMG_SIZE, IMG_SIZE),
    Normalize(),
])

TRAIN_TRANSFORM_FN = [
    Augmentor(
        dict_key="image",
        augment_fn=lambda x: AUG_TRAIN(image=x)["image"]),
]

INFER_TRANSFORM_FN = [
    Augmentor(
        dict_key="image",
        augment_fn=lambda x: AUG_INFER(image=x)["image"]),
    Augmentor(
        dict_key="image",
        augment_fn=lambda x: torch.tensor(x).permute(2, 0, 1)),
]


# ---- Data ----

class DataSource(AbstractDataSource):

    @staticmethod
    def prepare_transforms(*, mode, stage=None):
        if mode == "train":
            if stage in ["debug", "stage1"]:
                return transforms.Compose(
                    TRAIN_TRANSFORM_FN + INFER_TRANSFORM_FN)
            elif stage == "stage2":
                return transforms.Compose(INFER_TRANSFORM_FN)
        elif mode == "valid":
            return transforms.Compose(INFER_TRANSFORM_FN)
        elif mode == "infer":
            return transforms.Compose(INFER_TRANSFORM_FN)

    @staticmethod
    def prepare_loaders(args, data_params, stage=None):
        loaders = collections.OrderedDict()

        df, df_train, df_valid, df_infer = parse_in_csvs(data_params)

        open_fn = [
            ImageReader(
                row_key="filepath", dict_key="image",
                datapath=data_params.get("datapath", None)),
            ScalarReader(
                row_key="class", dict_key="target",
                default_value=-1, dtype=np.int64)
        ]
        open_fn = ReaderCompose(readers=open_fn)

        if len(df_train) > 0:
            labels = [x["class"] for x in df_train]
            sampler = BalanceClassSampler(labels, mode="upsampling")

            train_loader = UtilsFactory.create_loader(
                data_source=df_train,
                open_fn=open_fn,
                dict_transform=DataSource.prepare_transforms(
                    mode="train", stage=stage),
                dataset_cache_prob=getattr(args, "dataset_cache_prob", -1),
                batch_size=args.batch_size,
                workers=args.workers,
                shuffle=sampler is None,
                sampler=sampler)

            print("Train samples", len(train_loader) * args.batch_size)
            print("Train batches", len(train_loader))
            loaders["train"] = train_loader

        if len(df_valid) > 0:
            sampler = None

            valid_loader = UtilsFactory.create_loader(
                data_source=df_valid,
                open_fn=open_fn,
                dict_transform=DataSource.prepare_transforms(
                    mode="valid", stage=stage),
                dataset_cache_prob=-1,
                batch_size=args.batch_size,
                workers=args.workers,
                shuffle=False,
                sampler=sampler)

            print("Valid samples", len(valid_loader) * args.batch_size)
            print("Valid batches", len(valid_loader))
            loaders["valid"] = valid_loader

        if len(df_infer) > 0:
            infer_loader = UtilsFactory.create_loader(
                data_source=df_infer,
                open_fn=open_fn,
                dict_transform=DataSource.prepare_transforms(
                    mode="infer", stage=None),
                dataset_cache_prob=-1,
                batch_size=args.batch_size,
                workers=args.workers,
                shuffle=False,
                sampler=None)

            print("Infer samples", len(infer_loader) * args.batch_size)
            print("Infer batches", len(infer_loader))
            loaders["infer"] = infer_loader

        return loaders
