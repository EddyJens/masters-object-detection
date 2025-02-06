from monai.transforms import ScaleIntensityRanged
from generate_transforms import (
    generate_detection_train_transform,
    generate_detection_val_transform,
)
from monai.data import load_decathlon_datalist, Dataset, DataLoader
from monai.data.utils import no_collation
import torch

def load_data(
    gt_box_mode, patch_size, batch_size, amp, data_list_file_path,
    data_base_dir, a_min, a_max
):

    amp = True
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32
    
    ## ref windowing: https://www.kaggle.com/code/bardiakh/monai-io-windowing-overlay-saving
    intensity_transform = ScaleIntensityRanged(
        keys=["image"],
        a_min=a_min,
        a_max=a_max,
        b_min=0.0,
        b_max=1.0,
        clip=True
    )

    ### train
    train_transforms = generate_detection_train_transform(
        "image",
        "box",
        "label",
        gt_box_mode,
        intensity_transform,
        patch_size,
        batch_size,
        affine_lps_to_ras=True,
        amp=amp
    )

    train_data = load_decathlon_datalist(
        data_list_file_path,
        is_segmentation=True,
        data_list_key="training",
        base_dir=data_base_dir
    )

    train_ds = Dataset(
        data=train_data[: int(0.95 * len(train_data))],
        transform=train_transforms
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation
    )

    ### val
    val_transforms = generate_detection_val_transform(
        "image",
        "box",
        "label",
        gt_box_mode,
        intensity_transform,
        affine_lps_to_ras=True,
        amp=amp,
    )

    val_ds = Dataset(
        data=train_data[int(0.95 * len(train_data)) :],
        transform=val_transforms
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
    )

    return train_loader ,val_loader, len(train_ds)
