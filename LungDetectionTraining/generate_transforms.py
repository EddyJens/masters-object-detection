import torch
import numpy as np
from monai.transforms import (
    Compose,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated,
    BoxToMaskd,
    ClipBoxToImaged,
    ConvertBoxToStandardModed,
    MaskToBoxd,
    RandCropBoxByPosNegLabeld, # RandCropByPosNegLabeld
    RandFlipBoxd,
    RandRotateBox90d,
    RandZoomBoxd,
    StandardizeEmptyBoxd,
)

def generate_detection_train_transform(
    image_key,
    box_key,
    label_key,
    gt_box_mode,
    intensity_transform,
    patch_size,
    batch_size,
    affine_lps_to_ras=False,
    amp=True
):
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    return Compose(
        [
            LoadImaged(keys=[image_key], meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=torch.long),
            StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
            Orientationd(keys=[image_key], axcodes="RAS"),
            intensity_transform,
            EnsureTyped(keys=[image_key], dtype=torch.float16),
            ConvertBoxToStandardModed(box_keys=[box_key], mode=gt_box_mode),
            AffineBoxToImageCoordinated(
                box_keys=[box_key],
                box_ref_image_keys=image_key,
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            RandCropBoxByPosNegLabeld(
                image_keys=[image_key],
                box_keys=box_key,
                label_keys=label_key,
                spatial_size=patch_size,
                whole_box=True,
                num_samples=batch_size,
                pos=1,
                neg=1,
            ),
            RandZoomBoxd(
                image_keys=[image_key],
                box_keys=[box_key],
                box_ref_image_keys=[image_key],
                prob=0.2,
                min_zoom=0.7,
                max_zoom=1.4,
                padding_mode="constant",
                keep_size=True,
            ),
            ClipBoxToImaged(
                box_keys=box_key,
                label_keys=[label_key],
                box_ref_image_keys=image_key,
                remove_empty=True,
            ),
            RandFlipBoxd(
                image_keys=[image_key],
                box_keys=[box_key],
                box_ref_image_keys=[image_key],
                prob=0.5,
                spatial_axis=0,
            ),
            RandFlipBoxd(
                image_keys=[image_key],
                box_keys=[box_key],
                box_ref_image_keys=[image_key],
                prob=0.5,
                spatial_axis=1,
            ),
            RandFlipBoxd(
                image_keys=[image_key],
                box_keys=[box_key],
                box_ref_image_keys=[image_key],
                prob=0.5,
                spatial_axis=2,
            ),
            RandRotateBox90d(
                image_keys=[image_key],
                box_keys=[box_key],
                box_ref_image_keys=[image_key],
                prob=0.75,
                max_k=3,
                spatial_axes=(0, 1),
            ),
            BoxToMaskd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                box_ref_image_keys=image_key,
                min_fg_label=0,
                ellipse_mask=True,
            ),
            RandRotated(
                keys=[image_key, "box_mask"],
                mode=["nearest", "nearest"],
                prob=0.2,
                range_x=np.pi / 6,
                range_y=np.pi / 6,
                range_z=np.pi / 6,
                keep_size=True,
                padding_mode="zeros",
            ),
            MaskToBoxd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                min_fg_label=0,
            ),
            DeleteItemsd(keys=["box_mask"]),
            RandGaussianNoised(keys=[image_key], prob=0.1, mean=0, std=0.1),
            RandGaussianSmoothd(
                keys=[image_key],
                prob=0.1,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            RandScaleIntensityd(keys=[image_key], prob=0.15, factors=0.25),
            RandShiftIntensityd(keys=[image_key], prob=0.15, offsets=0.1),
            RandAdjustContrastd(keys=[image_key], prob=0.3, gamma=(0.7, 1.5)),
            EnsureTyped(keys=[image_key, box_key], dtype=compute_dtype),
            EnsureTyped(keys=[label_key], dtype=torch.long),
        ]
    )

def generate_detection_val_transform(
    image_key,
    box_key,
    label_key,
    gt_box_mode,
    intensity_transform,
    affine_lps_to_ras=False,
    amp=True,
):
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    val_transforms = Compose(
        [
            LoadImaged(keys=[image_key], meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=torch.long),
            StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
            Orientationd(keys=[image_key], axcodes="RAS"),
            intensity_transform,
            ConvertBoxToStandardModed(box_keys=[box_key], mode=gt_box_mode),
            AffineBoxToImageCoordinated(
                box_keys=[box_key],
                box_ref_image_keys=image_key,
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            EnsureTyped(keys=[image_key, box_key], dtype=compute_dtype),
            EnsureTyped(keys=label_key, dtype=torch.long),
        ]
    )
    return val_transforms


