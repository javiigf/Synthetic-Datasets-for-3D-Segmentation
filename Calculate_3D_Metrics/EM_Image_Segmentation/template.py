# Script based on 3D_template.py


##########################
#   ARGS COMPROBATION    #
##########################

import argparse
parser = argparse.ArgumentParser(
    description="Template based of template/template.py")
parser.add_argument("base_work_dir",
                    help="Path to code base dir , i.e ~/DeepLearning_EM")
parser.add_argument("data_dir", help="Path to data base dir")
parser.add_argument("result_dir",
                    help="Path to where the resulting output of the job will "\
                    "be stored")
parser.add_argument("-id", "--job_id", "--id", help="Job identifier",
                    default="unknown_job")
parser.add_argument("-rid","--run_id", "--rid", help="Run number of the same job",
                    type=int, default=0)
parser.add_argument("-gpu","--gpu", dest="gpu_selected",
                    help="GPU number according to 'nvidia-smi' command",
                    required=True)
args = parser.parse_args()


##########################
#        PREAMBLE        #
##########################

import os
import sys
sys.path.insert(0, args.base_work_dir)

# Working dir
os.chdir(args.base_work_dir)

# Create job directory
os.makedirs(args.result_dir, exist_ok=True)

# Limit the number of threads
from util import limit_threads, set_seed, create_plots, store_history,\
                 TimeHistory, threshold_plots, save_img, \
                 calculate_3D_volume_prob_map, check_masks
limit_threads()

# Try to generate the results as reproducible as possible
set_seed(42)

crops_made = False
job_identifier = args.job_id + '_' + str(args.run_id)


##########################
#        IMPORTS         #
##########################

import datetime
import random
import numpy as np
import math
import time
import tensorflow as tf
from data_3D_manipulation import load_and_prepare_3D_data_v2,\
                                 merge_3D_data_with_overlap, \
                                 crop_3D_data_with_overlap
from generators.data_3D_generators import VoxelDataGenerator
from networks.unet_3d import U_Net_3D
from metrics import jaccard_index_numpy, voc_calculation
from tensorflow.keras.callbacks import EarlyStopping
from aux.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tqdm import tqdm
from tensorflow.keras.utils import plot_model
from post_processing import calculate_z_filtering, ensemble16_3d_predictions
from skimage.io import imsave


############
#  CHECKS  #
############

now = datetime.datetime.now()
print("Date : {}".format(now.strftime("%Y-%m-%d %H:%M:%S")))
print("Arguments: {}".format(args))
print("Python       : {}".format(sys.version.split('\n')[0]))
print("Numpy        : {}".format(np.__version__))
print("Keras        : {}".format(tf.keras.__version__))
print("Tensorflow   : {}".format(tf.__version__))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_selected;


##########################                                                      
#  EXPERIMENT VARIABLES  #
##########################

### Dataset variables
# Main dataset data/mask paths
train_path = os.path.join(args.data_dir, 'train', 'x')
train_mask_path = os.path.join(args.data_dir, 'train', 'y')
test_path = os.path.join(args.data_dir, 'test', 'x')
test_mask_path = os.path.join(args.data_dir, 'test', 'y')
# Percentage of the training data used as validation                            
perc_used_as_val = 0.1
# Create the validation data with random images of the training data. If False
# the validation data will be the last portion of training images.
random_val_data = False


### Data shape: (x, y, z, channels) 
img_train_shape = (1024, 1024, 91, 1)
img_test_shape = (1024, 1024, 91, 1)

### 3D volume variables: (x, y, z, channels)
# Train shape of the 3D subvolumes
train_3d_desired_shape = (256, 256, 40, 1)
# Train shape of the 3D subvolumes
test_3d_desired_shape = (256, 256, 40, 1)
# Make overlap on train data                                                    
ov_train = True
# Wheter to use the rest of the train data when there is no exact division between
# it and the subvolume shape needed (train_3d_desired_shape). Only has sense when 
# ov_train is False                                                             
use_rest_train = False
# Percentage of overlap in (x, y, z). Set to 0 to calculate the minimun overlap 
overlap = (0,0,0)


### Data augmentation (DA) variables. Based on https://github.com/aleju/imgaug
# Flag to activate DA
da = True
# Create samples of the DA made. Useful to check the output images made.
aug_examples = True
# Flag to shuffle the training data on every epoch 
shuffle_train_data_each_epoch = True
# Flag to shuffle the validation data on every epoch
shuffle_val_data_each_epoch = False
# Histogram equalization
hist_eq = False
# Rotation of 90º to the subvolumes
rotation = True
# Flag to make flips on the subvolumes (horizontal and vertical)
flips = True
# Elastic transformations
elastic = False
# Gaussian blur
g_blur = False
# Gamma contrast 
gamma_contrast = False
# Flag to extract random subvolumnes during the DA
random_subvolumes_in_DA = False
# Calculate probability map to make random subvolumes to be extracted with high
# probability of having a mitochondria on the middle of it. Useful to avoid
# extracting a subvolume which less mitochondria information.
probability_map = False # Only active with random_subvolumes_in_DA
w_foreground = 0.94 # Only active with probability_map
w_background = 0.06 # Only active with probability_map


### Extra train data generation
# Number of times to duplicate the train data. Useful when 
# "random_subvolumes_in_DA" is made, as more original train data can be cover
replicate_train = 0


### Load previously generated model weigths
# Flag to activate the load of a previous training weigths instead of train 
# the network again
load_previous_weights = False
# ID of the previous experiment to load the weigths from 
previous_job_weights = args.job_id
# Prefix of the files where the weights are stored/loaded from
weight_files_prefix = 'model.fibsem_'
# Wheter to find the best learning rate plot. If this options is selected the 
# training will stop when 5 epochs are done
use_LRFinder = False


### Experiment main parameters
# Batch size value
batch_size_value = 2
# Optimizer to use. Possible values: "sgd" or "adam"
optimizer = "adam"
# Learning rate used by the optimization method
learning_rate_value = 0.0001
# Number of epochs to train the network
epochs_value = 100
# Number of epochs to stop the training process after no improvement
patience = 20


### Network architecture specific parameters
# Number of feature maps on each level of the network
feature_maps = [28, 36, 48, 64]
# Depth of the network
depth = 3
# Flag to activate the Spatial Dropout instead of use the "normal" dropout layer
spatial_dropout = False
# Values to make the dropout with. It's dimension must be equal depth+1. Set to
# 0 to prevent dropout
dropout_values = [0, 0, 0, 0]
# Flag to active batch normalization
batch_normalization = False
# Kernel type to use on convolution layers
kernel_init = 'he_normal'
# Activation function to use
activation = "elu"
# Downsampling to be made in Z. This value will be the third integer of the
# MaxPooling operation. When facing anysotropic datasets set it to get better
# performance
z_down = 1
# Number of classes. To generate data with more than 1 channel custom DA need to
# be selected. It can be 1 or 2.                                                                   
n_classes = 1
# Adjust the metric used accordingly to the number of clases. This code is planned 
# to be used in a binary classification problem, so the function 'jaccard_index_softmax' 
# will only calculate the IoU for the foreground class (channel 1)              
metric = "jaccard_index_softmax" if n_classes > 1 else "jaccard_index" 
# To take only the last class of the predictions, which corresponds to the
# foreground in a binary problem. If n_classes > 2 this should be disabled to
# ensure all classes are preserved
last_class = True if n_classes <= 2 else False


### Paths of the results                                             
# Directory where predicted images of the segmentation will be stored
result_dir = os.path.join(args.result_dir, 'results', job_identifier)

# per-image directories
result_bin_dir_per_image = os.path.join(result_dir, 'per_image_binarized')
result_no_bin_dir_per_image = os.path.join(result_dir, 'per_image_no_binarized')
ens_bin_dir_per_image = os.path.join(result_dir, 'per_image_ensemble')
ens_no_bin_dir_per_image = os.path.join(result_dir, 'per_image_ensemble_no_bin')
zfil_dir_per_image = os.path.join(result_dir, 'per_image_zfil')
ens_zfil_dir_per_image = os.path.join(result_dir, 'per_image_ens_zfil')

# 50% overlap directories 
result_bin_dir_50ov = os.path.join(result_dir, '50ov_binarized')
result_no_bin_dir_50ov = os.path.join(result_dir, '50ov_no_binarized')
ens_bin_dir_50ov = os.path.join(result_dir, '50ov_8ensemble_binarized')         
ens_no_bin_dir_50ov = os.path.join(result_dir, '50ov_8ensemble_no_binarized')   
ens_zfil_dir_50ov = os.path.join(result_dir, '50ov_8ensemble_zfil')             

# Full image directories                                                        
result_bin_dir_full = os.path.join(result_dir, 'full_binarized')
result_no_bin_dir_full = os.path.join(result_dir, 'full_no_binarized')
ens_bin_dir_full = os.path.join(result_dir, 'full_8ensemble')
ens_no_bin_dir_full = os.path.join(result_dir, 'full_8ensemble')
zfil_dir_full = os.path.join(result_dir, 'full_zfil')

# Name of the folder where the charts of the loss and metrics values while
# training the network will be shown. This folder will be created under the
# folder pointed by "args.base_work_dir" variable
char_dir = os.path.join(result_dir, 'charts')
# Directory where weight maps will be stored
loss_weight_dir = os.path.join(result_dir, 'loss_weights', args.job_id)
# Folder where smaples of DA will be stored
da_samples_dir = os.path.join(result_dir, 'aug')
# Folder where crop samples will be stored
check_crop_path = os.path.join(result_dir, 'check_crop')
# Name of the folder where weights files will be stored/loaded from. This folder
# must be located inside the directory pointed by "args.base_work_dir" variable.
# If there is no such directory, it will be created for the first time
h5_dir = os.path.join(args.result_dir, 'h5_files')
# Name of the folder to store the probability map to avoid recalculating it on
# every run
prob_map_dir = os.path.join(args.result_dir, 'prob_map')


### Callbacks
# To measure the time
time_callback = TimeHistory()
# Stop early and restore the best model weights when finished the training
earlystopper = EarlyStopping(
    patience=patience, verbose=1, restore_best_weights=True)
# Save the best model into a h5 file in case one need again the weights learned
os.makedirs(h5_dir, exist_ok=True)
checkpointer = ModelCheckpoint(
    os.path.join(h5_dir, weight_files_prefix + job_identifier + '.h5'),
    verbose=1, save_best_only=True)


print("###################\n"
      "#  SANITY CHECKS  #\n"
      "###################\n")

#check_masks(train_mask_path)
#check_masks(test_mask_path)


print("###############\n"
      "#  LOAD DATA  #\n"
      "###############\n")

X_train, Y_train, X_val,\
Y_val, X_test, Y_test,\
orig_test_shape, crop_test_shapes,\
filenames = load_and_prepare_3D_data_v2(
    train_path, train_mask_path, test_path, test_mask_path, img_train_shape,
    img_test_shape, val_split=perc_used_as_val, create_val=True,
    shuffle_val=random_val_data, random_subvolumes_in_DA=random_subvolumes_in_DA,
    test_subvol_shape=test_3d_desired_shape,
    train_subvol_shape=train_3d_desired_shape, ov=overlap)


print("###########################\n"
      "#  EXTRA DATA GENERATION  #\n"
      "###########################\n")

# Calculate the steps_per_epoch value to train in case
if replicate_train != 0:
    steps_per_epoch_value = int((replicate_train*X_train.shape[0])/batch_size_value)
    print("Data doubled by {} ; Steps per epoch = {}".format(replicate_train,
          steps_per_epoch_value))
else:
    steps_per_epoch_value = int(X_train.shape[0]/batch_size_value)


print("#######################\n"
      "#  DATA AUGMENTATION  #\n"
      "#######################\n")

# Calculate the probability map per image
train_prob = None
if probability_map == True:
    prob_map_file = os.path.join(prob_map_dir, 'prob_map.npy')
    if os.path.exists(prob_map_dir):
        train_prob = np.load(prob_map_file)
    else:
        train_prob = calculate_3D_volume_prob_map(
            Y_train, w_foreground, w_background, save_file=prob_map_file)

print("Preparing validation data generator . . .")
val_generator = VoxelDataGenerator(
    X_val, Y_val, random_subvolumes_in_DA=random_subvolumes_in_DA,
    subvol_shape=train_3d_desired_shape,
    shuffle_each_epoch=shuffle_val_data_each_epoch, batch_size=batch_size_value,
    da=False, n_classes=n_classes, val=True)
del X_val, Y_val

print("Preparing train data generator . . .")
train_generator = VoxelDataGenerator(                                           
    X_train, Y_train, random_subvolumes_in_DA=random_subvolumes_in_DA,          
    subvol_shape=train_3d_desired_shape,                                        
    shuffle_each_epoch=shuffle_train_data_each_epoch,                           
    batch_size=batch_size_value, da=da, hist_eq=hist_eq, flip=flips,            
    rotation=rotation, elastic=elastic, g_blur=g_blur,                          
    gamma_contrast=gamma_contrast, n_classes=n_classes, prob_map=train_prob,    
    extra_data_factor=replicate_train) 
del X_train, Y_train

# Create the test data generator without DA
print("Preparing test data generator . . .")
test_generator = VoxelDataGenerator(
    X_test, Y_test, random_subvolumes_in_DA=False, shuffle_each_epoch=False,
    batch_size=batch_size_value, da=False, n_classes=n_classes)

# Generate examples of data augmentation
if aug_examples == True:
    train_generator.get_transformed_samples(
        5, random_images=False, save_to_dir=True, out_dir=da_samples_dir)


print("#################################\n"
      "#  BUILD AND TRAIN THE NETWORK  #\n"
      "#################################\n")

print("Creating the network . . .")
model = U_Net_3D(train_3d_desired_shape, activation=activation, depth=depth,
                 feature_maps=feature_maps, drop_values=dropout_values,
                 spatial_dropout=spatial_dropout, batch_norm=batch_normalization,
                 k_init=kernel_init, optimizer=optimizer, lr=learning_rate_value,
                 n_classes=n_classes, z_down=z_down)

# Check the network created
model.summary(line_length=150)
os.makedirs(char_dir, exist_ok=True)
model_name = os.path.join(char_dir, "model_plot_" + job_identifier + ".png")
plot_model(model, to_file=model_name, show_shapes=True, show_layer_names=True)

h5_file=os.path.join(h5_dir, weight_files_prefix + previous_job_weights     
                     + '_' + str(args.run_id) + '.h5')

if load_previous_weights == False:
    results = model.fit(x=train_generator, validation_data=val_generator,
        validation_steps=len(val_generator), 
        steps_per_epoch=steps_per_epoch_value, epochs=epochs_value,
        callbacks=[earlystopper, checkpointer, time_callback])

print("Loading model weights from h5_file: {}".format(h5_file))
model.load_weights(h5_file)


print("################################\n"
      "#  PREPARE DATA FOR INFERENCE  #\n"
      "################################\n")

# Prepare test data for its use
if np.max(Y_test) > n_classes:
    Y_test = Y_test.astype('float32')
    Y_test *= 1./255
if np.max(X_test) > 2:
    X_test = X_test.astype('float32')
    X_test *= 1./255


print("##########################\n"
      "#  INFERENCE (per crop)  #\n"
      "##########################\n")

# Evaluate to obtain the loss value and the Jaccard index
print("Evaluating test data . . .")
score_per_crop = model.evaluate(test_generator, verbose=1)
loss_per_crop = score_per_crop[0]
jac_per_crop = score_per_crop[1]

print("Making the predictions on test data . . .")
preds_test = model.predict(test_generator, verbose=1)

# Take only the foreground class                                                
if n_classes > 1:
    preds_test = np.expand_dims(preds_test[...,1], -1)


print("#############################################\n"
      "#  Metrics (per image, merging subvolumes)  #\n"
      "#############################################\n")

# Merge the volumes to the original 3D images 
jac = 0
voc = 0
index = 0
for i in tqdm(range(len(orig_test_shape))):
    original_3d_shape = orig_test_shape[i]
    crop_3d_shape = crop_test_shapes[i]
    f_name = filenames[1][i]
    
    print("Original shape is {}".format(original_3d_shape))
    orig_preds_test, orig_Y_test = merge_3D_data_with_overlap(
        preds_test[index:index+crop_3d_shape[0]], original_3d_shape, 
        data_mask=Y_test[index:index+crop_3d_shape[0]], overlap=overlap, 
        verbose=True)
    orig_preds_test = orig_preds_test.astype(np.float32)
    orig_Y_test = orig_Y_test.astype(np.float32)

    print("Saving predicted images . . .")                                          
    os.makedirs(result_bin_dir_per_image, exist_ok=True)
    imsave(os.path.join(result_bin_dir_per_image, f_name+'.tiff'),
           (orig_preds_test> 0.5).astype(np.uint8))
    os.makedirs(result_no_bin_dir_per_image, exist_ok=True)
    imsave(os.path.join(result_no_bin_dir_per_image, f_name+'.tiff'), 
           orig_preds_test)
                                                                                    
    print("Calculate the Jaccard of the image 3D")
    j = jaccard_index_numpy(orig_Y_test, (orig_preds_test > 0.5).astype(np.uint8))
    v = voc_calculation(orig_Y_test, (orig_preds_test > 0.5).astype(np.uint8), j)
    print("Image {} ; IoU: {} ; VOC: {}".format(i,j,v))
    jac += j
    voc += v

    index += crop_3d_shape[0]
del orig_preds_test

print("Calculate metrics (per image) . . .")                                                
jac_per_image = jac/len(orig_test_shape)
voc_per_image = voc/len(orig_test_shape)
det_per_image = -1

print("~~~~ 16-Ensemble (per image) ~~~~")                                     
Y_test_ensemble = np.zeros(X_test.shape, dtype=np.float32)                        
for i in tqdm(range(X_test.shape[0])):                                          
    predictions_ensemble = ensemble16_3d_predictions(X_test[i],                       
        pred_func=(lambda img_batch_subdiv: model.predict(img_batch_subdiv)),
        n_classes=n_classes, last_class=last_class)   
    Y_test_ensemble[i] = predictions_ensemble
                                                                                
# Merge the volumes to the original 3D images
jac = 0
voc = 0
jac_z = 0
voc_z = 0
jac_s_z = 0
voc_s_z = 0
index = 0
for i in tqdm(range(len(orig_test_shape))):
    original_3d_shape = orig_test_shape[i]
    crop_3d_shape = crop_test_shapes[i]    
    f_name = filenames[1][i]

    orig_preds_test, orig_Y_test = merge_3D_data_with_overlap(
        Y_test_ensemble[index:index+crop_3d_shape[0]], original_3d_shape,
        data_mask=Y_test[index:index+crop_3d_shape[0]],
        overlap=overlap, verbose=False)
    orig_preds_test = orig_preds_test.astype(np.float32)
    orig_Y_test = orig_Y_test.astype(np.float32)

    print("Saving predicted images . . .")
    os.makedirs(ens_no_bin_dir_per_image, exist_ok=True)
    imsave(os.path.join(ens_no_bin_dir_per_image, f_name+'.tiff'),
           (orig_preds_test> 0.5).astype(np.uint8))
    os.makedirs(ens_bin_dir_per_image, exist_ok=True)
    imsave(os.path.join(ens_bin_dir_per_image, f_name+'.tiff'),
           orig_preds_test)

    print("Calculate metrics (ensemble + per subvolume). . .")
    j = jaccard_index_numpy(orig_Y_test, (orig_preds_test > 0.5).astype(np.uint8))
    v = voc_calculation(orig_Y_test, (orig_preds_test > 0.5).astype(np.uint8), j)
    print("Image {} ; IoU: {} ; VOC: {}".format(i,j,v))
    jac += j
    voc += v

    print("~~~~ Z-Filtering (per image) ~~~~")
    zfil_preds_test = calculate_z_filtering(orig_preds_test)
    zfil_preds_test = zfil_preds_test.astype(np.float32)

    print("Saving Z-filtered images . . .")
    os.makedirs(zfil_dir_per_image, exist_ok=True)
    imsave(os.path.join(zfil_dir_per_image, f_name+'.tiff'),
           (zfil_preds_test> 0.5).astype(np.uint8))

    print("Calculate metrics (Z-filtering + per crop) . . .")
    j = jaccard_index_numpy(orig_Y_test, (zfil_preds_test > 0.5).astype(np.uint8))
    v = voc_calculation(orig_Y_test, (zfil_preds_test > 0.5).astype(np.uint8), j)
    print("Image {} ; IoU: {} ; VOC: {}".format(i,j,v))
    jac_z += j
    voc_z += v

    print("~~~~ Ensemble + Z-Filtering (per subvolume) ~~~~")
    ens_zfil_preds_test = calculate_z_filtering(orig_preds_test)
    ens_zfil_preds_test = ens_zfil_preds_test.astype(np.float32)
    
    print("Saving ensembleed + Z-filtered images . . .")
    os.makedirs(ens_zfil_dir_per_image, exist_ok=True)
    imsave(os.path.join(ens_zfil_dir_per_image, f_name+'.tiff'),
           (ens_zfil_preds_test > 0.5).astype(np.uint8))

    print("Calculate metrics (Ensemble + Z-filtering per crop) . . .")
    j = jaccard_index_numpy(orig_Y_test, (ens_zfil_preds_test > 0.5).astype(np.uint8))
    v = voc_calculation(orig_Y_test, (ens_zfil_preds_test > 0.5).astype(np.uint8), j)
    print("Image {} ; IoU: {} ; VOC: {}".format(i,j,v))
    jac_s_z += j
    voc_s_z += v

    index += crop_3d_shape[0]
del orig_preds_test, zfil_preds_test, ens_zfil_preds_test

ens_jac_per_image = jac/len(orig_test_shape)
ens_voc_per_image = voc/len(orig_test_shape)
                                                                                
zfil_jac_per_image = jac_z/len(orig_test_shape)
zfil_voc_per_image = voc_z/len(orig_test_shape)
                                                                                
ens_zfil_jac_per_image = jac_s_z/len(orig_test_shape)
ens_zfil_voc_per_image = voc_s_z/len(orig_test_shape)
                                        

print("############################################################\n"
      "#  Metrics (per image, merging crops with 50% of overlap)  #\n"
      "############################################################\n")

jac = 0
voc = 0
index = 0
for i in tqdm(range(len(orig_test_shape))):
    original_3d_shape = orig_test_shape[i]
    crop_3d_shape = crop_test_shapes[i]    
    f_name = filenames[1][i]

    orig_X_test, orig_Y_test = merge_3D_data_with_overlap(
        X_test[index:index+crop_3d_shape[0]], original_3d_shape,
        data_mask=Y_test[index:index+crop_3d_shape[0]],
        overlap=overlap, verbose=True)

    orig_X_test = crop_3D_data_with_overlap(
        orig_X_test, test_3d_desired_shape, overlap=(0.5,0.5,0.5), verbose=True)

    Y_test_50ov = model.predict(orig_X_test, batch_size=batch_size_value, verbose=1) 

    # Take only the foreground class                                                
    if n_classes > 1:                                                               
        Y_test_50ov = np.expand_dims(Y_test_50ov[...,1], -1)                          
 
    Y_test_50ov = merge_3D_data_with_overlap(
        Y_test_50ov, original_3d_shape, overlap=(0.5,0.5,0.5), verbose=True)
    Y_test_50ov = Y_test_50ov.astype(np.float32)
    
    print("Saving 50% overlap predicted images . . .")
    os.makedirs(result_bin_dir_50ov, exist_ok=True)
    imsave(os.path.join(result_bin_dir_50ov, f_name+'.tiff'),
           (Y_test_50ov> 0.5).astype(np.uint8))
    os.makedirs(result_no_bin_dir_50ov, exist_ok=True)
    imsave(os.path.join(result_no_bin_dir_50ov, f_name+'.tiff'), Y_test_50ov)

    print("Calculate metrics (50% overlap) . . .")
    j = jaccard_index_numpy(orig_Y_test, (Y_test_50ov > 0.5).astype(np.uint8)) 
    v = voc_calculation(orig_Y_test, (Y_test_50ov > 0.5).astype(np.uint8), j)
    print("Image {} ; IoU: {} ; VOC: {}".format(i,j,v))
    jac += j
    voc += v

    index += crop_3d_shape[0]

del orig_X_test, Y_test_50ov

jac_50ov = j/len(orig_test_shape)
voc_50ov = v/len(orig_test_shape)
det_50ov = -1                                                               

print("~~~~ 16-Ensemble ~~~~")                                      
jac = 0
voc = 0
jac_z = 0
voc_z = 0
index = 0
for i in tqdm(range(len(orig_test_shape))):
    original_3d_shape = orig_test_shape[i]
    crop_3d_shape = crop_test_shapes[i]    
    f_name = filenames[1][i]

    orig_X_test, orig_Y_test = merge_3D_data_with_overlap(                      
        X_test[index:index+crop_3d_shape[0]], original_3d_shape,                
        data_mask=Y_test[index:index+crop_3d_shape[0]],                         
        overlap=overlap, verbose=True)                                          

    orig_X_test = crop_3D_data_with_overlap(                                    
        orig_X_test, test_3d_desired_shape, overlap=(0.5,0.5,0.5), verbose=True)

    Y_test_50ov_ensemble = np.zeros(orig_X_test.shape, dtype=np.float32)
    for j in tqdm(range(orig_X_test.shape[0])):
        predictions_ensembled = ensemble16_3d_predictions(orig_X_test[j],
            pred_func=(lambda img_batch_subdiv: model.predict(img_batch_subdiv)),   
            n_classes=n_classes, last_class=last_class)
        Y_test_50ov_ensemble[j] = predictions_ensembled

    # Take only the foreground class                                                
    if n_classes > 1:                                                           
        Y_test_50ov_ensemble = np.expand_dims(Y_test_50ov_ensemble[...,1], -1)                    
                                                                                
    orig_preds_test = merge_3D_data_with_overlap(                                   
        Y_test_50ov_ensemble, original_3d_shape, overlap=(0.5,0.5,0.5), verbose=True)    
    orig_preds_test = orig_preds_test.astype(np.float32)    
    
    print("Saving 50% overlap predicted images . . .")
    os.makedirs(ens_bin_dir_50ov, exist_ok=True)
    imsave(os.path.join(ens_bin_dir_50ov, f_name+'.tiff'),
           (orig_preds_test> 0.5).astype(np.uint8))
    os.makedirs(ens_no_bin_dir_50ov, exist_ok=True)
    imsave(os.path.join(ens_no_bin_dir_50ov, f_name+'.tiff'),
           orig_preds_test)

    print("Calculate metrics (50% overlap) . . .")
    j = jaccard_index_numpy(orig_Y_test, (orig_preds_test > 0.5).astype(np.uint8))
    v = voc_calculation(orig_Y_test, (orig_preds_test > 0.5).astype(np.uint8), j)
    print("Image {} ; IoU: {} ; VOC: {}".format(i,j,v))
    jac += j
    voc += v

    print("~~~~ Z-Filtering (50% overlap) ~~~~")
    zfil_preds_test = calculate_z_filtering(orig_preds_test)
    zfil_preds_test = zfil_preds_test.astype(np.float32)

    print("Saving Z-filtered images . . .")
    os.makedirs(ens_zfil_dir_50ov, exist_ok=True)
    imsave(os.path.join(ens_zfil_dir_50ov, f_name+'.tiff'), zfil_preds_test)

    print("Calculate metrics (Z-filtering + 50% overlap) . . .")
    j = jaccard_index_numpy(orig_Y_test, (zfil_preds_test > 0.5).astype(np.uint8))
    v = voc_calculation(orig_Y_test, (zfil_preds_test > 0.5).astype(np.uint8), j)
    print("Image {} ; IoU: {} ; VOC: {}".format(i,j,v))
    jac_z += j
    voc_z += v

    index += crop_3d_shape[0]

del orig_preds_test, orig_Y_test, zfil_preds_test, Y_test_50ov_ensemble

ens_jac_50ov = jac/len(orig_test_shape)
ens_voc_50ov = voc/len(orig_test_shape)

ens_zfil_jac_50ov = jac_z/len(orig_test_shape)
ens_zfil_voc_50ov = voc_z/len(orig_test_shape)


print("########################\n"
      "# Metrics (full image) #\n"
      "########################\n")

jac_full = -1
voc_full = -1
det_full = -1

ens_jac_full = -1
ens_voc_full = -1

zfil_jac_full = -1
zfil_voc_full = -1


print("####################################\n"
      "#  PRINT AND SAVE SCORES OBTAINED  #\n"
      "####################################\n")

if load_previous_weights == False:
    print("Epoch average time: {}".format(np.mean(time_callback.times)))
    print("Epoch number: {}".format(len(results.history['val_loss'])))
    print("Train time (s): {}".format(np.sum(time_callback.times)))
    print("Train loss: {}".format(np.min(results.history['loss'])))
    print("Train IoU: {}".format(np.max(results.history[metric])))
    print("Validation loss: {}".format(np.min(results.history['val_loss'])))
    print("Validation IoU: {}".format(np.max(results.history['val_'+metric])))

print("Test loss: {}".format(loss_per_crop))
print("Test IoU (per crop): {}".format(jac_per_crop))

print("Test IoU (merge into complete image): {}".format(jac_per_image))
print("Test VOC (merge into complete image): {}".format(voc_per_image))
print("Post-process: Ensemble - Test IoU (merge into complete image): {}".format(ens_jac_per_image))
print("Post-process: Ensemble - Test VOC (merge into complete image): {}".format(ens_voc_per_image))
print("Post-process: Z-Filtering - Test IoU (merge into complete image): {}".format(zfil_jac_per_image))
print("Post-process: Z-Filtering - Test VOC (merge into complete image): {}".format(zfil_voc_per_image))
print("Post-process: Ensemble + Z-Filtering - Test IoU (merge into complete image): {}".format(ens_zfil_jac_per_image))
print("Post-process: Ensemble + Z-Filtering - Test VOC (merge into complete image): {}".format(ens_zfil_voc_per_image))

print("Test IoU (merge with 50% overlap): {}".format(jac_50ov))
print("Test VOC (merge with 50% overlap): {}".format(voc_50ov))
print("Post-process: Ensemble - Test IoU (merge with 50% overlap): {}".format(ens_jac_50ov))
print("Post-process: Ensemble - Test VOC (merge with 50% overlap): {}".format(ens_voc_50ov))
print("Post-process: Ensemble + Z-Filtering - Test IoU (merge with 50% overlap): {}".format(ens_zfil_jac_50ov))
print("Post-process: Ensemble + Z-Filtering - Test VOC (merge with 50% overlap): {}".format(ens_zfil_voc_50ov))

print("Test IoU (full): {}".format(jac_full))
print("Test VOC (full): {}".format(voc_full))
print("Post-process: Ensemble - Test IoU (full): {}".format(ens_jac_full))
print("Post-process: Ensemble - Test VOC (full): {}".format(ens_voc_full))
print("Post-process: Ensemble + Z-Filtering - Test IoU (full): {}".format(zfil_jac_full))
print("Post-process: Ensemble + Z-Filtering - Test VOC (full): {}".format(zfil_voc_full))

if not load_previous_weights:
    scores = {}
    for name in dir():
        if not name.startswith('__') and ("_per_crop" in name or "_50ov" in name\
        or "_per_image" in name or "_full" in name):
            scores[name] = eval(name)

    create_plots(results, job_identifier, char_dir, metric=metric)

print("FINISHED JOB {} !!".format(job_identifier))
