import cv2
import os
import math
import statistics
import sys
import numpy as np
from skimage.io import imsave, imread
from utils.matching import match_using_VJI_and_PAI

predictedPath = "/media/pedro/6TB/jesus/methodology_naturalVariation/VJI/2dots/twodots_pred/"
#predictedPath = "/home/pedro/Escritorio/jesus/seg_compare/seg_compare/example_data/MARS"

groundTruthPath = "/media/pedro/6TB/jesus/methodology_naturalVariation/VJI/2dots/twodots_pred/"
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

fileName = "twodot.tif"

predictedLabels = imread(os.path.join(predictedPath, fileName));
groundTruthLabels = imread(os.path.join(groundTruthPath, fileName));

result = match_using_VJI_and_PAI(groundTruthLabels, predictedLabels)

print(result)
