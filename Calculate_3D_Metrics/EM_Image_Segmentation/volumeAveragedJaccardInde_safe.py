import cv2
import os
import math
import statistics
import sys
import numpy as np
from skimage.io import imsave, imread

predictedPath = "/media/pedro/6TB/jesus/methodology_naturalVariation/VJI/oldExps/pred/"
#predictedPath = "/home/pedro/Escritorio/jesus/seg_compare/seg_compare/example_data/MARS"

groundTruthPath = "/media/pedro/6TB/jesus/methodology_naturalVariation/VJI/oldExps/gt/"
#groundTruthPath = "/home/pedro/Escritorio/jesus/seg_compare/seg_compare/example_data/MARS"
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

fileName = "10d.1B.10.1.tif"
#fileName = "groud_truth_segmentation.tif"
#fileName2 = "MARS_segmentation.tif"
predictedLabels = imread(os.path.join(predictedPath, fileName));
groundTruthLabels = imread(os.path.join(groundTruthPath, fileName));

groundTruthCells = np.unique(groundTruthLabels)
predictedCells = np.unique(predictedLabels)

groundTruthLabelsNum = np.size(groundTruthCells)
predictedCellsNum = np.size(predictedCells)

#Initialize variables
jaccardIndex = np.zeros([groundTruthLabelsNum, predictedCellsNum])
groundTruthCellVolume = np.zeros([groundTruthLabelsNum])
predictedCellVolume = np.zeros([predictedCellsNum])
assymetricInclusionGC = np.zeros([groundTruthLabelsNum, predictedCellsNum])
assymetricInclusionPC = np.zeros([predictedCellsNum, groundTruthLabelsNum])
A = np.zeros([groundTruthLabelsNum], dtype=np.int)
B = np.zeros([groundTruthLabelsNum], dtype=np.int)
Bprime = np.zeros([predictedCellsNum], dtype=np.int)

#Relabel
newLabel = 0
for groundTruthCell in groundTruthCells:
    groundTruthLabels[groundTruthLabels==groundTruthCell] = newLabel
    newLabel = newLabel + 1
groundTruthCells = np.unique(groundTruthLabels)

newLabel = 0
for predictedCell in predictedCells:
    predictedLabels[predictedLabels==predictedCell] = newLabel
    newLabel = newLabel + 1
predictedCells = np.unique(predictedLabels)

#Calculate unions, intersections, jaccardIndex, volumes and assymetric inclusion indices
for predictedCell in predictedCells:
    #Calculate GroundTruth cell and predictedCellVolume Volume
    predictedCellVolume[predictedCell] = np.count_nonzero(predictedLabels==predictedCell)
matchingCells = np.unique(predictedLabels[(groundTruthLabels==3) & (predictedLabels>0)])
jaccardIndexSum = 0
interSum = 0
for groundTruthCell in groundTruthCells:

    gtCell = groundTruthLabels==groundTruthCell
    validSlices = [np.count_nonzero(gtCell[slice, :, :])>0 for slice in range(np.shape(gtCell)[0])]
    gtCell = gtCell[validSlices, :, :]
    
    matchingCells = np.unique(predictedLabels[(groundTruthLabels==groundTruthCell) & (predictedLabels>0)])
    

    #Calculate GroundTruth cell and predictedCellVolume Volume
    groundTruthCellVolume[groundTruthCell] = np.count_nonzero(groundTruthLabels==groundTruthCells[groundTruthCell])
    #print("CYST {} {}/{} matchingCells {}".format(fileName, groundTruthCell, groundTruthLabelsNum,matchingCells ))
    for predictedCell in matchingCells:

        pCell =  predictedLabels==predictedCell
        pCell = pCell[validSlices, :, :]
        
        #Calculate union and intersection
        intersection = (gtCell & pCell).sum()
        interSum = intersection +interSum
        union = groundTruthCellVolume[groundTruthCell]+predictedCellVolume[predictedCell]-intersection
        #union = gtCell.sum()+pCell.sum()
        
        #Calculate Jaccard Index (1)
        jaccardIndex[groundTruthCell, predictedCell] = intersection/union
        jaccardIndexSum = jaccardIndexSum + jaccardIndex[groundTruthCell, predictedCell]
        #Calculate Assymetric Inclusion Index (2)
        assymetricInclusionGC[groundTruthCell, predictedCell] = intersection/groundTruthCellVolume[groundTruthCell]
        assymetricInclusionPC[predictedCell, groundTruthCell] = intersection/predictedCellVolume[predictedCell]

    #Calculate A (3) and B (4)
    A[groundTruthCell] = np.argmax(jaccardIndex[groundTruthCell, :])
    B[groundTruthCell] = np.argmax(assymetricInclusionGC[groundTruthCell, :])

#Calculate B prime (5)
for predictedCell in predictedCells:
    Bprime[predictedCell] = np.argmax(assymetricInclusionPC[predictedCell, :])

#Calculate Volume Averaged Jaccard (VJI) (6)
upperPart = 0
lowerPart = 0

for groundTruthCell in groundTruthCells[1:]:
    upperPart = upperPart + groundTruthCellVolume[groundTruthCell]*jaccardIndex[groundTruthCell,predictedCells[A[groundTruthCell]]]
    lowerPart = lowerPart + groundTruthCellVolume[groundTruthCell]

VJI = upperPart/lowerPart

print('VJI = {}'.format(VJI))

#Under and over segmentation rates
pairAssociatedIndices = [0, 0];
for groundTruthCell in groundTruthCells:
    for predictedCell in predictedCells:
	    if(B[groundTruthCell]==predictedCell or Bprime[predictedCell]==groundTruthCell):
	        pairAssociatedIndices = np.vstack([pairAssociatedIndices, [groundTruthCell, predictedCell]])

print(pairAssociatedIndices)
pairAssociatedIndices = pairAssociatedIndices[2:]
#0 bijection | 1 Background | 2 Missed | 3 Invalid | 4 Oversegmentation | 5 Undersegmentation |
cellClassification = {}
for groundTruthCell in groundTruthCells[1:]:
    if Bprime[B[groundTruthCell]] == groundTruthCell and sum(Bprime==groundTruthCell)==1 and sum(B==B[groundTruthCell])==1:
        cellClassification[groundTruthCell] = 'bijection'
    elif B[groundTruthCell] == 0:
        cellClassification[groundTruthCell] = 'background'
    elif Bprime[B[groundTruthCell]] == 0:
        cellClassification[groundTruthCell] = 'missed'
    elif sum(Bprime==groundTruthCell)>1 and B[groundTruthCell]!=0:
        cellClassification[groundTruthCell] = 'oversegmented'
    elif sum(B==B[groundTruthCell])>1 and Bprime[B[groundTruthCell]]!=0 and sum(Bprime==groundTruthCell)<=1:
        cellClassification[groundTruthCell] = 'undersegmented'
    else:
        cellClassification[groundTruthCell] = 'missed'

for predictedCell in predictedCells[1:]:
    if Bprime[predictedCell] == 0 or cellClassification[Bprime[predictedCell]]=='missed':
        cellClassification[str(predictedCell) + '_pred'] = 'invalid'

invalid = sum(classification == 'invalid' for classification in cellClassification.values())
missed = sum(classification == 'missed' for classification in cellClassification.values())
background = sum(classification == 'background' for classification in cellClassification.values())
bijection = sum(classification == 'bijection' for classification in cellClassification.values())
oversegmented = sum(classification == 'oversegmented' for classification in cellClassification.values())
undersegmented = sum(classification == 'undersegmented' for classification in cellClassification.values())

validCells = len(cellClassification.values())
print(cellClassification)
print('valid cells: {}'.format(validCells))
print('background rate: {}'.format(background/validCells))
print('oversegmentation rate: {}'.format((oversegmented)/validCells))
print('undersegmentation rate: {}'.format((undersegmented)/validCells))
print('missed rate: {}'.format((missed)/validCells))
print('invalid rate: {}'.format((invalid)/validCells))
print('bijection rate: {}'.format(bijection/validCells))


    
