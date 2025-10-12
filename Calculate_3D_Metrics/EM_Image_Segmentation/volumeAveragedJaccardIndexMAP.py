import cv2
import os
import math
import statistics
import sys
import numpy as np
from skimage.io import imsave, imread

predictedPath = "/media/pedro/6TB/jesus/EM_Image_Segmentation/exp_results/cyst39_v2_BCM/results/cyst39_v2_BCM_5/per_image_instances_voronoi"
predictedPath = "/home/pedro/Escritorio/jesus/seg_compare/seg_compare/example_data/MARS"

groundTruthPath = "/media/pedro/6TB/jesus/methodology_naturalVariation/dataset_DANI/treeOrganization/test/y"
groundTruthPath = "/home/pedro/Escritorio/jesus/seg_compare/seg_compare/example_data/MARS"
 #   """Calcualte Volume Averaged Index (VGI) metric as well as over-segmentation and 
 #   under-segmentation rates based on the paper entitled "Assessment of deep learning algorithms
 #    for 3D instance segmentation of confocal image datasets" by A. Kar et al 
 #    (https://doi.org/10.1101/2021.06.09.447748)
 #    
 #    The numbers next to the calculations correspond to equation number in the aforementioned
 #    paper.                                                             
#
#       Inputs
#       ----------
#       predictedLabels : 4D Numpy array
#           Predicted data  E.g. ``(img_number, x, y, channels)``.##
#
#       groundTruthLabels : 4D Numpy array
 #          Ground Truth data. E.g. ``(img_number, x, y, channels)``.
#
 #      Returns
  #     -------

   # """
import time

fileName = "MARS_segmentation.tif"
fileName2 = "groud_truth_segmentation.tif"
predictedLabels = imread(os.path.join(predictedPath, fileName));
groundTruthLabels = imread(os.path.join(groundTruthPath, fileName2));

groundTruthCells = np.unique(groundTruthLabels)
predictedCells = np.unique(predictedLabels)

groundTruthLabelsNum = np.size(groundTruthCells)
predictedCellsNum = np.size(predictedCells)

#Initialize variables
union = np.zeros([groundTruthLabelsNum, predictedCellsNum])
intersection = np.zeros([groundTruthLabelsNum, predictedCellsNum])
jaccardIndex = np.zeros([groundTruthLabelsNum, predictedCellsNum])
groundTruthCellVolume = np.zeros([groundTruthLabelsNum])
predictedCellVolume = np.zeros([predictedCellsNum])
assymetricInclusionGC = np.zeros([groundTruthLabelsNum, predictedCellsNum])
assymetricInclusionPC = np.zeros([predictedCellsNum, groundTruthLabelsNum])
A = np.zeros([groundTruthLabelsNum], dtype=np.int)
B = np.zeros([groundTruthLabelsNum], dtype=np.int)
Bprime = np.zeros([predictedCellsNum], dtype=np.int)

#Calculate unions, intersections, jaccardIndex, volumes and assymetric inclusion indices

for groundTruthCell in range(groundTruthLabelsNum):
    print("CYST {} {}/{}".format(fileName, groundTruthCell, groundTruthLabelsNum))
    for predictedCell in range(predictedCellsNum):

        #Calculate union and intersection
        union[groundTruthCell, predictedCell] = np.count_nonzero((groundTruthLabels==groundTruthCells[groundTruthCell]) | (predictedLabels==predictedCells[predictedCell]))
        intersection[groundTruthCell, predictedCell] = np.count_nonzero((groundTruthLabels==groundTruthCells[groundTruthCell]) & (predictedLabels==predictedCells[predictedCell]))
        
        #Calculate Jaccard Index (1)
        jaccardIndex[groundTruthCell, predictedCell] = intersection[groundTruthCell, predictedCell]/union[groundTruthCell, predictedCell]
        
        #Calculate GroundTruth cell and predictedCellVolume Volume
        groundTruthCellVolume[groundTruthCell] = np.count_nonzero(groundTruthLabels==groundTruthCells[groundTruthCell])
        predictedCellVolume[predictedCell] = np.count_nonzero(predictedLabels==predictedCells[predictedCell])
        
        #Calculate Assymetric Inclusion Index (2)
        assymetricInclusionGC[groundTruthCell, predictedCell] = intersection[groundTruthCell, predictedCell]/groundTruthCellVolume[groundTruthCell]
        
        assymetricInclusionPC[predictedCell, groundTruthCell] = intersection[groundTruthCell, predictedCell]/predictedCellVolume[predictedCell]
        
    #Calculate A (3) and B (4)

    A[groundTruthCell] = np.argmax(jaccardIndex[groundTruthCell, :])
    B[groundTruthCell] = np.argmax(assymetricInclusionGC[groundTruthCell, :])
	

 
 
#Calculate B prime (5)
for predictedCell in range(predictedCellsNum):
    Bprime[predictedCell] = np.argmax(jaccardIndex[:, predictedCell])

#Calculate Volume Averaged Jaccard (VJI) (6)
upperPart = 0
lowerPart = 0
for groundTruthCell in range(2, groundTruthLabelsNum):
    upperPart = groundTruthCellVolume[groundTruthCell]*jaccardIndex[groundTruthCell,A[groundTruthCell]]
    lowerPart = groundTruthCellVolume[groundTruthCell]
    
VJI = upperPart/lowerPart

print('VJI = {}'.format(VJI))

#Under and over segmentation rates
pairAssociatedIndices = [0, 0];
for groundTruthCell in range(groundTruthLabelsNum):
    for predictedCell in range(predictedCellsNum):
	    if(B[groundTruthCell]==predictedCell or Bprime[predictedCell]==groundTruthCell):
	        pairAssociatedIndices = np.vstack([pairAssociatedIndices, [groundTruthCell, predictedCell]])

oversegmented = 0;
for groundTruthCell in range(groundTruthLabelsNum):
    if(np.shape(pairAssociatedIndices[pairAssociatedIndices[:, 0]==groundTruthCell, :])[0]>1):
        for predictedCell in pairAssociatedIndices[pairAssociatedIndices[:, 0]==groundTruthCell, 1]:
            if Bprime[predictedCell] not in pairAssociatedIndices[pairAssociatedIndices[:, 0]==groundTruthCell, 0]:
                break
            else:
                oversegmented = oversegmented + 1 
                
undersegmented = 0;
for predictedCell in range(predictedCellsNum):
    if(np.shape(pairAssociatedIndices[pairAssociatedIndices[:, 1]==predictedCell, :])[0]>1):
        for groundTruthCell in pairAssociatedIndices[pairAssociatedIndices[:, 1]==predictedCell, 0]:
            if B[groundTruthCell] not in pairAssociatedIndices[pairAssociatedIndices[:, 1]==predictedCell, 0]:
                break
            else:
                undersegmented = undersegmented + 1
 
print('oversegmentation rate: {}'.format(oversegmented/groundTruthLabelsNum))
print('undersegmentation rate: {}'.format(undersegmented/groundTruthLabelsNum))
    
