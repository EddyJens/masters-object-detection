import numpy as np
import SimpleITK as sitk
import scipy

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

def gen_coordinate_and_mask(img):

    x, y, z = img.shape

    coordinates = []
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if img[i][j][k] == 1:
                    coordinates.append((i, j, k))
                    
    xs = []
    ys = []
    zs = []
    for a in range(len(coordinates)):
        xs.append(coordinates[a][0])
        ys.append(coordinates[a][1])
        zs.append(coordinates[a][2])

    axes = [x, y, z]
    nodule_mask = np.zeros(axes)
    for i in range(np.min(xs), np.max(xs)):
        for j in range(np.min(ys), np.max(ys)):
            for k in range(np.min(zs), np.max(zs)):
                nodule_mask[i][j][k] = 1
    
    return {
        "nodule_mask": nodule_mask,
        "x_min": np.min(xs),
        "y_min": np.min(ys),
        "z_min": np.min(zs),
        "x_max": np.max(xs),
        "y_max": np.max(ys),
        "z_max": np.max(zs)
    }



























    


