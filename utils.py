import SimpleITK as sitk
import numpy as np

def normalize_planes(npzarray):
    # maxHU = 400.
    maxHU = 1500.
    # minHU = -1000.
    minHU = -600.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    image_array = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return image_array, origin, spacing

