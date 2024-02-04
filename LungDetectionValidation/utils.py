"""
helpful functions used by multiple scripts
"""
import numpy as np
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt
from monai.metrics import compute_froc_curve_data, compute_froc_score

def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    image_array = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return image_array, origin, spacing

def world_to_pixel(origin, spacing, world_coord_cent, world_coord_diam):
    voxel_coord_diam = world_coord_diam / spacing
    voxel_coord_cent = ((world_coord_cent / -1) + origin)/spacing

    voxel_coord_cent[2] = voxel_coord_cent[2] / -1

    voxel_min = voxel_coord_cent - voxel_coord_diam
    voxel_max = voxel_coord_cent + voxel_coord_diam

    return voxel_min, voxel_max

def any_match(prediction, target):
    """
    verify if there is any match in the 3D arrays
    """
    match = True
    for i in range(0, 3):
        a = range(prediction['voxel_min'][i], prediction['voxel_max'][i])
        b = range(target['voxel_min'][i], target['voxel_max'][i])
        if set(a).isdisjoint(b):
           match = False

    return match

def calculate_volume(poligon):
    a1 = poligon['voxel_max'][0] - poligon['voxel_min'][0]
    a2 = poligon['voxel_max'][1] - poligon['voxel_min'][1]
    a3 = poligon['voxel_max'][2] - poligon['voxel_min'][2]

    return a1 * a2 * a3

def calc_intersect_volume(poligon1, poligon2):
    a1 = min(
        poligon1['voxel_max'][0], poligon2['voxel_max'][0]) - max(
        poligon1['voxel_min'][0], poligon2['voxel_min'][0])
    a2 = min(
        poligon1['voxel_max'][1], poligon2['voxel_max'][1]) - max(
        poligon1['voxel_min'][1], poligon2['voxel_min'][1])
    a3 = min(
        poligon1['voxel_max'][2], poligon2['voxel_max'][2]) - max(
        poligon1['voxel_min'][2], poligon2['voxel_min'][2])

    return a1 * a2 * a3

def calc_iou(target, prediction):
    ref_vol = calculate_volume(target)
    found_vol = calculate_volume(prediction)  
    
    vol_of_overlap = calc_intersect_volume(prediction, target)
    vol_of_union = ref_vol + found_vol - vol_of_overlap

    return vol_of_overlap/vol_of_union

def vanilla_nms_pytorch(P: torch.tensor, thresh_iou: float):
    """
    Vanilla implementation
    ref: https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    """
    x1 = P[:, 0]
    y1 = P[:, 1]
    z1 = P[:, 2]
    x2 = P[:, 3]
    y2 = P[:, 4]
    z2 = P[:, 5]

    scores = P[:, 6]
    volumes = (x2 - x1) * (y2 - y1) * (z2 - z1)
    order = scores.argsort()
    keep = []

    while len(order) > 0:
        idx = order[-1]
        keep.append(P[idx])
        order = order[:-1]

        if len(order) == 0:
            break

        xx1 = torch.index_select(x1,dim = 0, index = order)
        xx2 = torch.index_select(x2,dim = 0, index = order)
        yy1 = torch.index_select(y1,dim = 0, index = order)
        yy2 = torch.index_select(y2,dim = 0, index = order)
        zz1 = torch.index_select(z1,dim = 0, index = order)
        zz2 = torch.index_select(z2,dim = 0, index = order)

        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        zz1 = torch.max(zz1, z1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])
        zz2 = torch.min(zz2, z2[idx])

        w = xx2 - xx1
        h = yy2 - yy1
        d = zz2 - zz1

        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        d = torch.clamp(d, min=0.0)

        inter = w*h*d
        rem_volumes = torch.index_select(volumes, dim=0, index=order) 
        union = (rem_volumes - inter) + volumes[idx]
        IoU = inter / union
        mask = IoU < thresh_iou
        order = order[mask]
    
    return keep

def gen_froc_plot(all_fps_per_image, all_total_sensitivity):
    _, ax = plt.subplots()

    plt.plot(all_fps_per_image[0], all_total_sensitivity[0]*100, color='red', label='a_min=-1000.0, a_max=-200.0')
    plt.plot(all_fps_per_image[1], all_total_sensitivity[1]*100, color='blue', label='a_min=-1000.0, a_max=0')
    plt.plot(all_fps_per_image[2], all_total_sensitivity[2]*100, color='green', label='a_min=-1000.0, a_max=400.0')
    plt.plot(all_fps_per_image[3], all_total_sensitivity[3]*100, color='orange', label='a_min=-1000.0, a_max=200.0')
    plt.plot(all_fps_per_image[4], all_total_sensitivity[4]*100, color='black', label='all')
    plt.xlabel('FP/exame')
    plt.ylabel('Sensibilidade (%)')
    plt.grid(True, linestyle='dashed')
    plt.legend()
    plt.xlim(0, 8)
    # plt.ylim(75, 100)
    a = np.arange(7)
    ax.xaxis.set_ticks(a)
    ax.xaxis.set_ticklabels(['1/8', '2/8', '4/8', 1, 2, 4, 8])

def gen_precision_recall_plot(precision_recall_curve, all_testy, all_lr_probs):
    lr_precision, lr_recall, _ = precision_recall_curve(all_testy[0], all_lr_probs[0])
    lr_precision1, lr_recall1, _ = precision_recall_curve(all_testy[1], all_lr_probs[1])
    lr_precision2, lr_recall2, _ = precision_recall_curve(all_testy[2], all_lr_probs[2])
    lr_precision3, lr_recall3, _ = precision_recall_curve(all_testy[3], all_lr_probs[3])
    lr_precision4, lr_recall4, _ = precision_recall_curve(all_testy[4], all_lr_probs[4])
    plt.plot(lr_recall, lr_precision, color='red', label='a_min=-1000.0, a_max=-200.0')
    plt.plot(lr_recall1, lr_precision1, color='blue', label='a_min=-1000.0, a_max=0')
    plt.plot(lr_recall2, lr_precision2, color='green', label='a_min=-1000.0, a_max=400.0')
    plt.plot(lr_recall3, lr_precision3, color='orange', label='a_min=-1000.0, a_max=200.0')
    plt.plot(lr_recall4, lr_precision4, color='black', label='all')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

def gen_tabular_data(
        tp, fp, fn, fp_probs, tp_probs, num_images, num_targets
):
    precision = tp /(tp + fp)
    recall = tp/ (tp + fn)
    fps_per_image, total_sensitivity = compute_froc_curve_data(
        fp_probs=np.array(fp_probs),
        tp_probs=np.array(tp_probs),
        num_targets=num_targets,
        num_images=num_images
    )

    p1 = compute_froc_score(fps_per_image, total_sensitivity, eval_thresholds=(0.125))
    p2 = compute_froc_score(fps_per_image, total_sensitivity, eval_thresholds=(0.25))
    p3 = compute_froc_score(fps_per_image, total_sensitivity, eval_thresholds=(0.5))
    p4 = compute_froc_score(fps_per_image, total_sensitivity, eval_thresholds=(1))
    p5 = compute_froc_score(fps_per_image, total_sensitivity, eval_thresholds=(2))
    p6 = compute_froc_score(fps_per_image, total_sensitivity, eval_thresholds=(4))
    p7 = compute_froc_score(fps_per_image, total_sensitivity, eval_thresholds=(8))
    cpm = (p1 + p2 + p3 + p4 + p5 + p6 + p7)/7

    return (
        fps_per_image, total_sensitivity, precision,
        recall, p1, p2, p3, p4, p5, p6, p7, cpm
    )

def convert_to_tensor(pred_boxes):
    """
    convert voxel_min, max format to monai standard: xyzxyz
    """
    converted = []
    for a in pred_boxes:
        converted.append([a['voxel_min'][0], a['voxel_min'][1], a['voxel_min'][2], a['voxel_max'][0], a['voxel_max'][1], a['voxel_max'][2]])

    return torch.tensor(converted)
