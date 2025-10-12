#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:19:01 2024

@author: Jesús Á Andrés San-Román
"""

import skimage.io as skio
import numpy as np
from skimage.io import imsave
from skimage import morphology 
#from skimage.filters import threshold_otsu
#from skimage.filters import gaussian
from scipy.spatial.distance import cdist
from scipy import ndimage
#import math

def generateVoronoi(path, iterId):
    
    size = [100, 100, 100]
    
    nDots = 200
    
    tissueStartX = 10
    tissueStartY = 10
    tissueStartZ = 10
    
    tissueEndX = 90
    tissueEndY = 90
    tissueEndZ = 90
    
    #Create mask
    mask = np.zeros(size)
    mask[tissueStartX:tissueEndX, tissueStartY:tissueEndY, tissueStartZ:tissueEndZ] = 1
    
    tries = 1000
    seedCoords = []
    for tryIx in range(tries):
        proposedCoords = [np.random.randint(tissueStartX, tissueEndX), np.random.randint(tissueStartY, tissueEndY), np.random.randint(tissueStartZ, tissueEndZ)]
        if tryIx == 0:
             seedCoords.append(proposedCoords)
        elif tryIx != 0:
            if np.any(cdist(np.array([proposedCoords]), np.array(seedCoords))<10):
                continue
            else:
                 seedCoords.append(proposedCoords)
        shape = np.shape(seedCoords)
        if shape[0] > nDots-1:
            break
        if tryIx == 1000 and shape[0] < nDots:
            raise ValueError("Less min distance or less nCells needed")
            
    img = np.zeros(size)
    seedCoords = np.array(seedCoords)
    img[tuple(seedCoords.T)] = 1
    
    img = morphology.binary_dilation(img, morphology.ball(radius=2))
    
    img = ndimage.label(img)
    img, _ = img
    #
    voronoiChunk = img*mask
    
    binaryVoronoiChunk = (voronoiChunk > 0)*1
    
    binaryVoronoiChunk = binaryVoronoiChunk.astype('uint8')
    
    #Cell Perimeter
    erodedVoronoiChunk= morphology.binary_erosion(binaryVoronoiChunk, morphology.ball(radius=1))
    cellPerimeter = binaryVoronoiChunk - erodedVoronoiChunk
    
    #Save images
#    imsave('/media/pedro/6TB/jesus/methodology_naturalVariation/unet_dani/img.tif', img.astype(np.uint16))
#    imsave('/media/pedro/6TB/jesus/methodology_naturalVariation/unet_dani/mask.tif', cellPerimeter.astype(np.uint16))
    
    #Define ids to fill where there is mask but no labels
    idsToFill = np.argwhere((mask==1) & (img==0))
    labelPerId = np.zeros(np.size(idsToFill));
    
    idsPerim = np.argwhere(cellPerimeter==1)
    
    labelsPerimIds = voronoiChunk[cellPerimeter==1]
    
    #Generating voronoi
    for nId in range(1,len(idsToFill)):
        distCoord = cdist([idsToFill[nId]], idsPerim)
        idSeedMin = np.argwhere(distCoord==np.min(distCoord))
        idSeedMin = idSeedMin[0][1]
        labelPerId[nId] = labelsPerimIds[idSeedMin]
        voronoiChunk[idsToFill[nId][0], idsToFill[nId][1], idsToFill[nId][2]] = labelPerId[nId]
    
    for nId in range(1,len(idsToFill)):
        voronoiChunk[idsToFill[nId][0], idsToFill[nId][1], idsToFill[nId][2]] = labelPerId[nId]
    
    cellOutlines = np.zeros(size)
    
    for cellId in range(1,len(np.unique(img))):
        print(cellId)
        currentCellImg = (voronoiChunk==cellId)
        currentCellImgEroded = morphology.binary_erosion(currentCellImg, morphology.ball(radius=1))
        currentCellImgOutline = np.double(currentCellImg)-currentCellImgEroded
        cellOutlines[currentCellImgOutline==1] = 255;
    
    #Save img
    imsave(path+str(iterId)+'.tif', cellOutlines.astype(np.uint8))


numVoronois = 100;
for iterIx in range(numVoronois):
    print(iterIx)
    try:
        generateVoronoi('/media/pedro/6TB/jesus/voronoiBasedGenerativeModel/voronoiPlanes/', iterIx)
    except:
        continue