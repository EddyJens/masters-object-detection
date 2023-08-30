import numpy as np
import SimpleITK as sitk
import scipy
from sklearn.neighbors import NearestNeighbors

def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    image_array = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return image_array, origin, spacing

def resample(image, previous_spacing, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array(previous_spacing, dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def gen_coordinate_and_masks(img):

    x, y, z = img.shape
    axes = [x, y, z]
    seg_refined = False

    ## nodule initial coordinates
    coordinates = []
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if img[i][j][k] == 1:
                    coordinates.append((i, j, k))

    ### nodule clean coordinates
    # find all distances
    X = np.array(coordinates)
    nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    # sum all distances found
    disto = []
    for index, dista in enumerate(distances):
        disto.append({
            "index": index,
            "distance": dista.sum()
        })
    # sort distances, to get the bigger ones
    sorted_disto = sorted(disto, key=lambda x:x["distance"], reverse=True)
    # create outliers list, based on a threshold
    outliers = []
    refined = []
    for unique_value in sorted_disto:
        if unique_value['distance'] > 1500:
            outliers.append(unique_value)
        else:
            refined.append(coordinates[unique_value["index"]])
    # generate outlier mask
    outlier_mask = np.zeros(axes)
    if len(outliers) > 0:
        seg_refined = True
        for distrib in outliers:
            a = coordinates[distrib["index"]][0]
            b = coordinates[distrib["index"]][1]
            c = coordinates[distrib["index"]][2]
            outlier_mask[a][b][c] = 1

    ## generate nodule mask coordinates
    xs = []
    ys = []
    zs = []
    for a in range(len(refined)):
        xs.append(refined[a][0])
        ys.append(refined[a][1])
        zs.append(refined[a][2])

    ## nodule mask bounding box
    nodule_bounding_box = np.zeros(axes)
    for i in range(np.min(xs), np.max(xs)+1):
        for j in range(np.min(ys), np.max(ys)+1):
            for k in range(np.min(zs), np.max(zs)+1):
                nodule_bounding_box[i][j][k] = 1
    
    return {
        "nodule_bounding_box": nodule_bounding_box,
        "outlier_mask": outlier_mask,
        "coordinates": refined,
        "exam_refined": seg_refined,
        "x_min": np.min(xs),
        "y_min": np.min(ys),
        "z_min": np.min(zs),
        "x_max": np.max(xs) + 1,
        "y_max": np.max(ys) + 1,
        "z_max": np.max(zs) + 1
    }






















    


