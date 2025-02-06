import SimpleITK as sitk
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, segmentation
import scipy.ndimage as ndimage
import numpy as np
import matplotlib
from tqdm import tqdm

def load_scan(path: str, meta: bool = False) -> np.ndarray:
    itkimage = sitk.ReadImage(path)

    if meta:
        origin = np.array(list(itkimage.GetOrigin()))
        spacing = np.array(list(itkimage.GetSpacing()))
        return sitk.GetArrayFromImage(itkimage), origin, spacing

    return sitk.GetArrayFromImage(itkimage)

def plot_3d(image, threshold=-300, name=None):
    p = image.transpose(2, 1, 0)

    verts, faces, normals, values = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    ax.view_init(20, 90)

    if name is None:
        plt.show()
    else:
        plt.savefig(name + '.png')

    plt.close()
    matplotlib.use('Agg')
    
    return True

def optimized_generate_markers(image):
    #Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    #Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    #Creation of the Watershed Marker matrix
    marker_watershed = np.zeros((512, 512), dtype=int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128
    
    return marker_internal, marker_watershed

def optimized_separate_lungs(image):
    #Creation of the markers as shown above:
    marker_internal, marker_watershed = optimized_generate_markers(image)
    
    #Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)
    
    #Watershed algorithm
    watershed = segmentation.watershed(sobel_gradient, marker_watershed)
    
    #Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3,3))
    outline = outline.astype(bool)
    
    #Performing Black-Tophat Morphology for reinclusion
    #Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    #Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)
    
    #Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    #Close holes in the lungfilter
    #fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)
    
    #Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    # segmented = np.where(lungfilter == 1, image, -2000*np.ones((512, 512)))

    ### added by me!! experiment to check if that way the nodules fit the lung mask
    # lungfilter = ndimage.morphology.binary_erosion(lungfilter, iterations = 15)
    # lungfilter = ndimage.morphology.binary_dilation(lungfilter, iterations = 30)
    
    return lungfilter * 1 # converting into 0 and 1

def segmented_entire_image(image):
    for i, axial_slice in tqdm(enumerate(image)):
        image[i] = optimized_separate_lungs(axial_slice)

    return image

# not used, because the current segment algorithm uses the HU values
def normalize_planes(npzarray):
    """
    used to generate 8bit image from 16bit (keep all visual information)
    """
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray
