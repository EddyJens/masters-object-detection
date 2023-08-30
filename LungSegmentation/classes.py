import numpy as np
import SimpleITK as sitk
from skimage.measure import regionprops
from configs import RESOURCES_PATH, MASK_PATH

class CTScan(object):
    def __init__(self, seriesuid, centers, radii, clazz):
        self._seriesuid = seriesuid
        self._centers = centers
        path = RESOURCES_PATH + '/' + self._seriesuid + '.mhd'
        self._ds = sitk.ReadImage(path)
        self._spacing = np.array(list(reversed(self._ds.GetSpacing())))
        self._origin = np.array(list(reversed(self._ds.GetOrigin())))
        self._radii = radii
        self._clazz = clazz
        mask = sitk.ReadImage(MASK_PATH + '/' + self._seriesuid+ '.nrrd') 
        self._mask = sitk.GetArrayFromImage(mask)

    def preprocess(self):
        # self._resample()
        # self._segment_lung_from_ct_scan()
        # self._normalize()
        # self._zero_center()
        self._change_coords()

    def _change_coords(self):
        new_coords = self._get_voxel_coords()
        self._centers = new_coords

    def _get_voxel_coords(self):
        voxel_coords = [self._get_world_to_voxel_coords(j) for j in range(len(self._centers))]
        return voxel_coords
    
    def _get_world_to_voxel_coords(self, idx):
        return tuple(self._world_to_voxel(self._centers[idx]))
    
    def _world_to_voxel(self, worldCoord):
        stretchedVoxelCoord = np.absolute(np.array(worldCoord) - np.array(self._origin))
        voxelCoord = stretchedVoxelCoord / np.array(self._spacing)
        return voxelCoord.astype(int)

    def get_info_dict(self):
        (min_z, min_y, min_x, max_z, max_y, max_x) = (None, None, None, None, None, None)
        for region in regionprops(self._mask):
            min_z, min_y, min_x, max_z, max_y, max_x = region.bbox
        assert (min_z, min_y, min_x, max_z, max_y, max_x) != (None, None, None, None, None, None)
        min_point = (min_z, min_y, min_x)
        max_point = (max_z, max_y, max_x)

        return {
            'seriesuid': self._seriesuid,
            'radii': self._radii,
            'centers': self._centers,
            'spacing': list(self._spacing),
            'lungs_bounding_box': [min_point, max_point],
            'class': self._clazz
        }

class PatchMaker(object):
    def __init__(self, seriesuid: str, coords: list, radii: list, spacing: list,
                lungs_bounding_box: list, file_path: str, clazz: int):
        self._seriesuid = seriesuid
        self._coords = coords
        self._spacing = spacing
        self._radii = radii
        self._image = np.load(file=f'{file_path}')
        self._clazz = clazz
        self._lungs_bounding_box = lungs_bounding_box

    
    