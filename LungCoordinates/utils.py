import ast
import numpy as np
import SimpleITK as sitk

def from_np_array(array_string):
    array_string = ','.join(array_string.replace('[  ', '[').replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))

def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    image_array = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return image_array, origin, spacing
