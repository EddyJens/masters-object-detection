import numpy as np
import SimpleITK as sitk
import torch

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

def nms_pytorch(P: torch.tensor, thresh_iou: float):
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
























        
