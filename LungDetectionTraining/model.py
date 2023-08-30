import torch
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.networks.nets import resnet
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor
)
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector


def load_model(
    returned_layers, base_anchor_shapes, conv1_t_stride, n_input_channels,
    spatial_dims, fg_labels, verbose, balanced_sampler_pos_fraction,
    score_thresh, nms_thresh, val_patch_size
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### build anchor generator
    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**l for l in range(len(returned_layers) + 1)],
        base_anchor_shapes=base_anchor_shapes,
    )

    ### build network
    conv1_t_size = [max(7, 2 * s + 1) for s in conv1_t_stride]
    backbone = resnet.ResNet(
        block=resnet.ResNetBottleneck,
        layers=[3, 4, 6, 3],
        block_inplanes=resnet.get_inplanes(),
        n_input_channels=n_input_channels,
        conv1_t_stride=conv1_t_stride,
        conv1_t_size=conv1_t_size,
    )
    feature_extractor = resnet_fpn_feature_extractor(
        backbone=backbone,
        spatial_dims=spatial_dims,
        pretrained_backbone=False,
        trainable_backbone_layers=None,
        returned_layers=returned_layers,
    )
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    size_divisible = [s * 2 * 2 ** max(returned_layers) for s in feature_extractor.body.conv1.stride]
    net = torch.jit.script(
        RetinaNet(
            spatial_dims=spatial_dims,
            num_classes=len(fg_labels),
            num_anchors=num_anchors,
            feature_extractor=feature_extractor,
            size_divisible=size_divisible,
        )
    )

    ### build detector
    detector = RetinaNetDetector(network=net, anchor_generator=anchor_generator, debug=verbose).to(device)

    #### set training components
    detector.set_atss_matcher(num_candidates=4, center_in_gt=False)
    detector.set_hard_negative_sampler(
        batch_size_per_image=64,
        positive_fraction=balanced_sampler_pos_fraction,
        pool_size=20,
        min_neg=16,
    )
    detector.set_target_keys(box_key="box", label_key="label")

    #### set validation components
    detector.set_box_selector_parameters(
        score_thresh=score_thresh,
        topk_candidates_per_level=1000,
        nms_thresh=nms_thresh,
        detections_per_img=100,
    )
    detector.set_sliding_window_inferer(
        roi_size=val_patch_size,
        overlap=0.25,
        sw_batch_size=1,
        mode="constant",
        device="cpu",
    )

    return detector, device






















    
    