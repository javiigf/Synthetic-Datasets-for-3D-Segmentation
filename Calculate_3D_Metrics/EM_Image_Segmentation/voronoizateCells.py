import skimage.io as skio
import numpy as np
import cv2
from skimage.io import imsave
from skimage import morphology 
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from scipy.spatial.distance import cdist
from scipy import ndimage

#Read img and mask
img = skio.imread('/media/pedro/6TB/jesus/methodology_naturalVariation/unet_dani/per_image_instances/4d.3B.5_1.tif')
mask = skio.imread('/media/pedro/6TB/jesus/methodology_naturalVariation/unet_dani/per_image_instances_BCM/4d.3B.5_1.tif')
mask = mask[:, :, :, 2]

#binarize
thresh = threshold_otsu(mask)
binaryMask = mask > thresh

#Close to fill holes
closedBinaryMask = morphology.closing(binaryMask, morphology.ball(radius=5)).astype(np.uint8)

#
voronoiCyst = img*closedBinaryMask

binaryVoronoiCyst = (voronoiCyst > 0)*1

binaryVoronoiCyst = binaryVoronoiCyst.astype('uint8')

#Cell Perimeter
erodedVoronoiCyst = morphology.binary_erosion(binaryVoronoiCyst, morphology.ball(radius=2))
cellPerimeter = binaryVoronoiCyst - erodedVoronoiCyst

#Save images
imsave('/media/pedro/6TB/jesus/methodology_naturalVariation/unet_dani/4d.3B.5_1_perimeter.tif', cellPerimeter.astype(np.uint16))
imsave('/media/pedro/6TB/jesus/methodology_naturalVariation/unet_dani/4d.3B.5_1_binVor.tif', binaryVoronoiCyst.astype(np.uint16))
imsave('/media/pedro/6TB/jesus/methodology_naturalVariation/unet_dani/4d.3B.5_1_closedBinaryMask.tif', closedBinaryMask.astype(np.uint16))

#Define ids to fill where there is mask but no labels
idsToFill = np.argwhere((closedBinaryMask==1) & (img==0))
labelPerId = np.zeros(np.size(idsToFill));

idsPerim = np.argwhere(cellPerimeter==1)

labelsPerimIds = voronoiCyst[cellPerimeter==1]


#Generating voronoi
for nId in range(1,len(idsToFill)):
    distCoord = cdist([idsToFill[nId]], idsPerim)
    idSeedMin = np.argwhere(distCoord==np.min(distCoord))
    idSeedMin = idSeedMin[0][1]
    labelPerId[nId] = labelsPerimIds[idSeedMin]
    voronoiCyst[idsToFill[nId][0], idsToFill[nId][1], idsToFill[nId][2]] = labelPerId[nId]

for nId in range(1,len(idsToFill)):
    voronoiCyst[idsToFill[nId][0], idsToFill[nId][1], idsToFill[nId][2]] = labelPerId[nId]

#Save img
imsave('/media/pedro/6TB/jesus/methodology_naturalVariation/unet_dani/4d.3B.5_1_VORONOIZATED.tif', voronoiCyst.astype(np.uint16))

