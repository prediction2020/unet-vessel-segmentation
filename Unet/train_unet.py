"""
File name: train_unet.py
Author: Jana Rieger
Date created: 06/24/2018

This is the main script for training the model. It loads the datasets, creates the unet model and trains it with the
loaded datasets. After training the model and results of training are saved to files.
"""

import csv

import numpy as np
import os
from Unet import config
from Unet.utils.prepare_train_val_sets import create_training_datasets
from Unet.utils.train_function import train_and_save

# fix random seed for reproducibility
np.random.seed(7)
# set numpy to print only 3 decimal digits for neatness
np.set_printoptions(precision=9, suppress=True)

# -----------------------------------------------------------
# MODEL PARAMETERS
# -----------------------------------------------------------

################################################
# SET PARAMETERS FOR TUNING
################################################
patch_size_list = [96]  # size of one patch n x n
batch_size_list = [8, 16, 32, 64]  # list with batch sizes
num_epochs = 10  # number of epochs
learning_rate_list = [1e-4, 1e-5]  # list with learning rates of the optimizer Adam
dropout_list = [0.0, 0.1, 0.2]  # percentage of weights to be dropped
threshold = config.THRESHOLD
factor_train_samples = 2  # how many times to augment training samples with the ImageDataGenerator per one epoch
rotation_range = 30  # for ImageDataGenerator
horizontal_flip = False  # for ImageDataGenerator
vertical_flip = True  # for ImageDataGenerator
shear_range = 20  # for ImageDataGenerator
width_shift_range = 0  # for ImageDataGenerator
height_shift_range = 0  # for ImageDataGenerator
fine_grid = True  # True for tuning hyperparameters in fine grid tuning, False for random rough grid
num_random_param_combinations = 0  # only necessary to set for random rough grid tuning
################################################

# general parameter
num_channels = config.NUM_CHANNELS
activation = config.ACTIVATION
final_activation = config.FINAL_ACTIVATION
loss = config.LOSS_FUNCTION
metrics = config.METRICS
optimizer = config.OPTIMIZER
num_patches = config.NUM_PATCHES
patients = config.PATIENTS

print('loss', loss)
print('metric', metrics)
print('patch size list', patch_size_list)
print('number of epochs', num_epochs)
print('batch size list', batch_size_list)
print('learning rate list', learning_rate_list)
print('dropout rate list', dropout_list)
print('threshold', threshold)
print('factor to multiply the train samples', factor_train_samples)
print('augmentation: rotation range', rotation_range)
print('augmentation: horizontal_flip', horizontal_flip)
print('augmentation: vertical_flip', vertical_flip)
print('augmentation: shear_range', shear_range)
print('augmentation: width_shift_range', width_shift_range)
print('augmentation: height_shift_range', height_shift_range)
print('fine_grid', fine_grid)
print('num_random_param_combinations', num_random_param_combinations)
print('________________________________________________________________________________')

if not os.path.exists(config.MODELS_DIR):
    os.makedirs(config.MODELS_DIR)

# open csv file for writing random tuned params and write the header
tuned_params_file = config.get_tuned_parameters()
header = ['patch size', 'num epochs', 'batch size', 'learning rate', 'dropout']
with open(tuned_params_file, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(header)

# TRAINING LOOPS
for patch_size in patch_size_list:
    # -----------------------------------------------------------
    # LOADING MODEL DATA
    # -----------------------------------------------------------
    train_X, train_y, val_X, val_y, mean, std = create_training_datasets(patch_size, num_patches,
                                                                         patients)

    for batch_size in batch_size_list:

        if fine_grid:
            # -----------------------------------------------------------
            # FINE GRID FOR PARAMETER TUNING
            # -----------------------------------------------------------
            for lr in learning_rate_list:
                for dropout in dropout_list:
                    # write the parameters to csv file
                    with open(tuned_params_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([patch_size, num_epochs, batch_size, lr, dropout])
                    # -----------------------------------------------------------
                    # TRAIN UNET AND SAVE MODEL AND RESULTS
                    # -----------------------------------------------------------
                    train_and_save(train_X, train_y, val_X, val_y,
                                   patch_size, num_epochs, batch_size, lr, dropout,
                                   num_channels, activation, final_activation, optimizer, loss, metrics, num_patches,
                                   factor_train_samples, mean, std, threshold,
                                   rotation_range, horizontal_flip, vertical_flip, shear_range,
                                   width_shift_range, height_shift_range)
        else:
            # -----------------------------------------------------------
            # RANDOM ROUGH GRID FOR PARAMETER TUNING
            # -----------------------------------------------------------
            lr_mesh, dropout_mesh = np.meshgrid(learning_rate_list, dropout_list)
            lr_mesh = lr_mesh.flatten()
            dropout_mesh = dropout_mesh.flatten()
            print(lr_mesh)
            print(dropout_mesh)
            random_inds = np.random.choice(len(lr_mesh), num_random_param_combinations, replace=False)
            print(random_inds)

            for i in random_inds:
                print('random index:', i)
                lr = lr_mesh[i]
                dropout = dropout_mesh[i]

                # write the parameters to csv file
                with open(tuned_params_file, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([patch_size, num_epochs, batch_size, lr, dropout])
                # -----------------------------------------------------------
                # TRAIN UNET AND SAVE MODEL AND RESULTS
                # -----------------------------------------------------------
                train_and_save(train_X, train_y, val_X, val_y,
                               patch_size, num_epochs, batch_size, lr, dropout,
                               num_channels, activation, final_activation, optimizer, loss, metrics, num_patches,
                               factor_train_samples, mean, std, threshold,
                               rotation_range, horizontal_flip, vertical_flip, shear_range,
                               width_shift_range, height_shift_range)

print('DONE')
