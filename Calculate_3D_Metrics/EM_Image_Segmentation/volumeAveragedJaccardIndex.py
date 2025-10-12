import cv2
import os
import math
import statistics
import sys
import numpy as np
from skimage.io import imsave, imread

predictedPath = "/media/pedro/6TB/jesus/methodology_naturalVariation/VJI/oldExps/pred/"
predictedPath = "/home/pedro/Escritorio/jesus/seg_compare/seg_compare/example_data/MARS"

groundTruthPath = "/media/pedro/6TB/jesus/methodology_naturalVariation/VJI/oldExps/gt/"
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

fileName = "10d.1B.10.1.tif"
fileName = "groud_truth_segmentation15_1.tif"
fileName2 = "MARS_segmentation15_1.tif"

#Read images
predictedLabels = imread(os.path.join(predictedPath, fileName2));
groundTruthLabels = imread(os.path.join(groundTruthPath, fileName));

#Unique cellIds
groundTruthCells = np.unique(groundTruthLabels)
predictedCells = np.unique(predictedLabels)

#Number of labels
groundTruthLabelsNum = np.size(groundTruthCells)
predictedCellsNum = np.size(predictedCells)

print(groundTruthLabelsNum)
#Background Threshold
#if a cell shares <50% of inclussion index w/ back it's not considered background
# even if the greatest inclussion index value of the cell is w/ the background
backgroundThreshold = 0.5

#Initialize variables
relabeledGT = np.zeros_like(groundTruthLabels)
relabeledPred = np.zeros_like(predictedLabels)
jaccardIndex = np.zeros([groundTruthLabelsNum, predictedCellsNum])
groundTruthCellVolume = np.zeros([groundTruthLabelsNum])
predictedCellVolume = np.zeros([predictedCellsNum])
assymetricInclusionGC = np.zeros([groundTruthLabelsNum, predictedCellsNum])
assymetricInclusionPC = np.zeros([predictedCellsNum, groundTruthLabelsNum])
A = np.zeros([groundTruthLabelsNum], dtype=np.int)
B = np.zeros([groundTruthLabelsNum], dtype=np.int)
Bprime = np.zeros([predictedCellsNum], dtype=np.int)

#Relabel (we assume that background is the lowest value in image)
newLabel = 0
for groundTruthCell in groundTruthCells:
    relabeledGT[groundTruthLabels==groundTruthCell] = newLabel
    newLabel = newLabel + 1
groundTruthLabels = relabeledGT
groundTruthCells = np.unique(groundTruthLabels)

newLabel = 0
for predictedCell in predictedCells:
    relabeledPred[predictedLabels==predictedCell] = newLabel
    newLabel = newLabel + 1
predictedLabels = relabeledPred
predictedCells = np.unique(predictedLabels)

#Calculate unions, intersections, jaccardIndex, volumes and assymetric inclusion indices
#VJI is not strictly neccessary, could be removed or maybe hidden w/ an if statement.
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

#Remove cells that are just slightly (background threshold) assigned to gt background 

for predictedCell in predictedCells[1:]:
    if Bprime[predictedCell] == 0 and max(assymetricInclusionPC[predictedCell, :])<backgroundThreshold:
        Bprime[predictedCell] = np.argsort(assymetricInclusionPC[predictedCell, :])[-2];

#Under and over segmentation rates
pairAssociatedIndices = [0, 0];
for groundTruthCell in groundTruthCells:
    for predictedCell in predictedCells:
	    if(B[groundTruthCell]==predictedCell or Bprime[predictedCell]==groundTruthCell):
	        pairAssociatedIndices = np.vstack([pairAssociatedIndices, [groundTruthCell, predictedCell]])

pairAssociatedIndices = pairAssociatedIndices[2:]
print(B)
print(Bprime)
# Explain classifications:
# bijection: one to one
# missed: gtCell to nothing
# background: predCell to nothing
# oversegentation: several predCells to one gtCell
# undersegmentation: one predCell to several gtCells

cellClassification = {}

for groundTruthCell in groundTruthCells[1:]:
    if Bprime[B[groundTruthCell]] == groundTruthCell and sum(Bprime==groundTruthCell)==1 and sum(B==B[groundTruthCell])==1:
        cellClassification[groundTruthCell] = 'bijection'
    elif B[groundTruthCell] == 0:
        cellClassification[groundTruthCell] = 'missed'
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
        if max(assymetricInclusionPC[predictedCell, :])>backgroundThreshold:
            cellClassification[str(predictedCell) + '_pred'] = 'background'


background = sum(classification == 'background' for classification in cellClassification.values())
missed = sum(classification == 'missed' for classification in cellClassification.values())
bijection = sum(classification == 'bijection' for classification in cellClassification.values())
oversegmented = sum(classification == 'oversegmented' for classification in cellClassification.values())
undersegmented = sum(classification == 'undersegmented' for classification in cellClassification.values())

#All code above is GT wise. This is the Pred wise part to match seg_compare results.

predWiseOversegmentation = 0
predWiseUndersegmentation = 0
predWiseUndersegmentedCells = list()

for element in cellClassification.keys():
    if cellClassification[element] == 'oversegmented':
        predWiseOversegmentation += sum(Bprime==element)
    elif cellClassification[element] == 'undersegmented':
        predWiseUndersegmentedCells.append(B[element])
predWiseUndersegmentation = len(np.unique(predWiseUndersegmentedCells))
       
       

print(cellClassification)

print('##############################')
print('#      GT wise results       #')
print('##############################')
#Valid cells: all gt cells + background cells
validCells = len(cellClassification.values())
print('number of cells: {}'.format(validCells))
print('correct segmentations rate: {}'.format(bijection/validCells))
print('oversegmentation rate: {}'.format((oversegmented)/validCells))
print('undersegmentation rate: {}'.format((undersegmented)/validCells))
print('background rate: {}'.format(background/validCells))
print('missing rate: {}'.format((missed)/validCells))


print('##############################')
print('#     Pred wise results      #')
print('##############################')
print('pred wise reslts')
predWiseValidCells = bijection + predWiseOversegmentation + predWiseUndersegmentation + missed + background
print('number of cells: {}'.format(predWiseValidCells))
print('correct segmentations rate: {}'.format(bijection/predWiseValidCells))
print('oversegmentation rate: {}'.format((predWiseOversegmentation)/predWiseValidCells))
print('undersegmentation rate: {}'.format((predWiseUndersegmentation)/predWiseValidCells))
print('background rate: {}'.format(background/predWiseValidCells))
print('missing rate: {}'.format((missed)/predWiseValidCells))

    
