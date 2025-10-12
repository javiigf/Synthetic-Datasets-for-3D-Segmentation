import cv2
import os
import math
import statistics
import sys
import numpy as np
import pandas as pd
from skimage.io import imsave, imread
import networkx as nx

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
BACKGROUND_LABEL = 0
thresh_background = 0.5
fileName = "10d.1B.10.1.tif"
fileName = "groud_truth_segmentation.tif"
fileName2 = "MARS_segmentation.tif"

#Read images
predictedLabels = imread(os.path.join(predictedPath, fileName2));
groundTruthLabels = imread(os.path.join(groundTruthPath, fileName));

#Unique cellIds
groundTruthCells = np.unique(groundTruthLabels)
predictedCells = np.unique(predictedLabels)

#Number of labels
groundTruthLabelsNum = np.size(groundTruthCells)
predictedCellsNum = np.size(predictedCells)

#Initialize variables
relabeledGT = np.zeros_like(groundTruthLabels)
relabeledPred = np.zeros_like(predictedLabels)
groundTruthCellVolume = np.zeros([groundTruthLabelsNum])
predictedCellVolume = np.zeros([predictedCellsNum])

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


df_target = pd.DataFrame(columns = ['target', 'reference', 'target_in_reference'])
df_reference = pd.DataFrame(columns = ['target', 'reference', 'reference_in_target'])

for groundTruthCell in groundTruthCells:

    gtCell = groundTruthLabels==groundTruthCell
    validSlices = [np.count_nonzero(gtCell[slice, :, :])>0 for slice in range(np.shape(gtCell)[0])]
    gtCell = gtCell[validSlices, :, :]
    
    #Calculate GroundTruth cell and predictedCellVolume Volume
    groundTruthCellVolume[groundTruthCell] = np.count_nonzero(groundTruthLabels==groundTruthCells[groundTruthCell])

    for predictedCell in predictedCells:

        pCell =  predictedLabels==predictedCell
        pCell = pCell[validSlices, :, :]
        
        #Calculate union and intersection
        intersection = (gtCell & pCell).sum()

        df_target.loc[len(df_target.index)] = [predictedCell, groundTruthCell, intersection/predictedCellVolume[predictedCell]]
        df_reference.loc[len(df_reference.index)] = [predictedCell, groundTruthCell, intersection/groundTruthCellVolume[groundTruthCell]]
    


### - Solve the background associations - ###
# target --> reference background
target_background = df_target.target.loc[(df_target.reference == BACKGROUND_LABEL) &
                                         (df_target.target_in_reference >= thresh_background)].to_list()

# reference --> target background
reference_background = df_reference.reference.loc[(df_reference.target == BACKGROUND_LABEL) &
                                                  (df_reference.reference_in_target >= thresh_background)].to_list()

# - Add the background (make sure we remove them for the clique associations)
target_background.append(BACKGROUND_LABEL)
reference_background.append(BACKGROUND_LABEL)

# - Remove the cells associated with the background + backgrounds
df_target = df_target.loc[~((df_target.target.isin(target_background)) |
                            (df_target.reference == BACKGROUND_LABEL))].copy()
df_reference = df_reference.loc[~((df_reference.reference.isin(reference_background)) |
                                  (df_reference.target == BACKGROUND_LABEL))].copy()
                       
### - Get the 1 <--> 1 associations - ###
# - Associate each reference/target cell with the target/reference cell in which it is the most included
df_target = df_target.loc[df_target.groupby('target')['target_in_reference'].idxmax()]
df_reference = df_reference.loc[df_reference.groupby('reference')['reference_in_target'].idxmax()]

# - Convert in dict
target_in_reference = df_target[['target', 'reference']].set_index('target').to_dict()['reference']
reference_in_target = df_reference[['reference', 'target']].set_index('reference').to_dict()['target']

print(df_reference)
print(df_target)

### - Build the associations (bijection, over-segmentation,...) - ###
# - Create a bipartite graph where nodes represent the A labels (left) and B labels (right)
#   and the edges the associations obtained using the max/min methods

# - Reindex the labels
target_labels = list(set(df_target.target.values) | set(df_reference.target.values))
reference_labels = list(set(df_target.reference.values) | set(df_reference.reference.values))

label_tp_list = [(m, 'l') for m in target_labels] + [(d, 'r') for d in reference_labels]
lg2nid = dict(zip(label_tp_list, range(len(label_tp_list))))

# - Create the graph
G = nx.Graph()

G.add_nodes_from([(nid, {'label': lab, 'group': g}) for (lab, g), nid in lg2nid.items()])

target_to_ref_list = [(lg2nid[(i, 'l')], lg2nid[(j, 'r')]) for i, j in target_in_reference.items()]
G.add_edges_from(target_to_ref_list)

ref_to_target_list = [(lg2nid[(i, 'r')], lg2nid[(j, 'l')]) for i, j in reference_in_target.items()]
G.add_edges_from(ref_to_target_list)

# - Overlap analysis
# - Get the  target_cells <--> reference_cells from the connected subgraph in G
connected_graph = [list(G.subgraph(c)) for c in nx.connected_components(G)]

# - Gather all the connected subgraph and reindex according to the image labels
nid2lg = {v: k for k, v in lg2nid.items()}

out_results = []
for c in connected_graph:
    if len(c) > 1:  # at least two labels
        target, reference = [], []
        for nid in c:
            if nid2lg[nid][1] == 'l':
                target.append(nid2lg[nid][0])  # label from target image
            else:
                reference.append(nid2lg[nid][0])  # label from reference image

        out_results.append({'target': target, 'reference': reference})

# - Add the background associations
# - target --> reference background
for lab in target_background:
    if lab != BACKGROUND_LABEL:  # ignore target background
        out_results.append({'target': [lab], 'reference': []})

# - reference --> target background
for lab in reference_background:
    if lab != BACKGROUND_LABEL:  # ignore reference background
        out_results.append({'target': [], 'reference': [lab]})

out_results = pd.DataFrame(out_results)

def segmentation_state(row):
    if len(row.reference) == 0:
        return 'background'
    elif len(row.target) == 0:
        return 'missing'
    elif len(row.target) == 1:
        if len(row.reference) == 1:
            return 'one-to-one'
        else:
            return 'under-segmentation'
    else:
        if len(row.reference) == 1:
            return 'over-segmentation'
        else:
            return 'misc.'
        
out_results['segmentation_state'] = out_results.apply(segmentation_state, axis=1) # add name for each type of association
print(out_results[:50])

ignore_background = True

cell_statistics = {'one-to-one': 0, 'over-segmentation': 0, 'under-segmentation': 0, 'misc.': 0}

if not ignore_background:
    cell_statistics['background'] = 0 # add background

state_target = {lab: state for list_lab, state
                in zip(out_results.target.values, out_results.segmentation_state.values)
                for lab in list_lab if state in cell_statistics}

for lab, state in state_target.items():
    cell_statistics[state] += 1

total_cells = len(state_target)
cell_statistics = {state: np.around(val / total_cells * 100, 2) for state, val in cell_statistics.items()}

# - add the missing  cells : percentage of reference cells that are missing in the predicted segmentation
total_reference_cells = len(np.unique([item for sublist in out_results.reference.values for item in sublist]))
missing_cells = [item for sublist in out_results.reference.loc[out_results.segmentation_state == 'missing'].values for item in sublist]

total_missing = len(missing_cells)
cell_statistics['missing'] = np.around(total_missing/total_reference_cells * 100, 2)

print("Number of cells:", total_cells)
print("% of correct segmentations :", cell_statistics['one-to-one'])
print("% of oversegmentation :", cell_statistics['over-segmentation'])
print("% of undersegmentation :", cell_statistics['under-segmentation'])
print("% of missing ground-truth cells :", cell_statistics['missing'])


