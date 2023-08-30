# Masters-object-detection
Masters 3D object detection implementation

### LungAnnFix

- folder with code to convert segmentation annotation to bounding box (considering the bigger body as the desired nodule)

### LungCoordinates

- folder with LUNA16 spacing and origin validation
- folder with scripts to convert voxel coordinates to world coordinates

### LungCoordinatesValidation

- TODO: complete with info

### LungDetectionTraining

- code to perform object 3D detection training

### LungDetectionValidation

- code to perform object 3D detection validation

### LungSegmentation

- code to perform lung segmentations (monai example, and the one that generated best results, with minimal time to process the data), with their visualization saving and mask saving. TODO: implement improvements to this algorithm!!
- contains reference code used by Lucas (in reference folder)

### LungSegmentationValidation

- code used to verify if all the bounding box annotations fit the lung segmentation (TODO: need to finish implementation)

### LungVisualizations

- TODO: move all visualization code to this folder!

### POCS

#### reference
- contains KNN reference code

#### torch-utils
- contains example code, of how to run an object detection model, into a real image (2D)
- contains code to verify GPU usage

#### statistics
- contains code to generate histograms of the exams dimmensions
- 

#### tutorial-to-object-detection
- contains tutorial to object detection (explanation of how it works) TODO: move to one note!

#### preprocessing -tutorial
- contains code to perform first version (vanila) of lung segmentation - execute fast, but return more errors
- contains code to perform watershed algorithm for lung segmentation (takes too long to generate files)
- contains a helper folder with examples to perform clahe, 3d-plot and matplitlib plot conversion to png

#### pedestrian-detection-mask-r-cnn
- mask-r-cnn example implementation, with pedestrian dataset (example from pytorch documentation, to perform object detection)











