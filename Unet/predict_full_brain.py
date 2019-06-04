"""
File name: predict_full_brain.py
Author: Jana Rieger
Date created: 03/12/2018

This is the main script for predicting a segmentation of an input MRA image. Segmentations can be predicted for multiple
models eather on rough grid (the parameters are then read out from the Unet/models/tuned_params.cvs file) or on fine
grid.
"""

import os
from Unet import config
from Unet.utils.helper import read_tuned_params_from_csv
from Unet.utils.predict_function import predict_and_save

################################################
# SET PARAMETERS FOR FINE GRID
################################################
dataset = 'test'  # train / val / set
patch_size_list = [96]  # list of sizes of one patch n x n
batch_size_list = [8, 16, 32, 64]  # list of batch sizes
num_epochs = 10  # number of epochs
learning_rate_list = [1e-4, 1e-5]  # list of learning rates of the optimizer Adam
dropout_list = [0.0, 0.1, 0.2]  # list of dropout rates: percentage of weights to be dropped
fine_grid = True  # True for tuning hyperparameters in fine grid tuning, False for random rough grid
################################################

num_patients_train = len(config.PATIENTS['train']['working']) + len(
    config.PATIENTS['train']['working_augmented'])  # number of patients in training category
num_patients_val = len(config.PATIENTS['val']['working']) + len(
    config.PATIENTS['val']['working_augmented'])  # number of patients in validation category
patients = config.PATIENTS[dataset]['working'] + config.PATIENTS[dataset]['working_augmented']

data_dirs = config.MODEL_DATA_DIRS
if not os.path.exists(config.RESULTS_DIR + dataset + '/'):
    os.makedirs(config.RESULTS_DIR + dataset + '/')
tuned_params_file = config.get_tuned_parameters()

# PARAMETER LOOPS
if fine_grid:
    # -----------------------------------------------------------
    # FINE GRID FOR PARAMETER TUNING
    # -----------------------------------------------------------
    for patch_size in patch_size_list:
        for batch_size in batch_size_list:
            for lr in learning_rate_list:
                for dropout in dropout_list:
                    for patient in patients:
                        predict_and_save(patch_size, num_epochs, batch_size, lr, dropout, patient, num_patients_train,
                                         num_patients_val, data_dirs, dataset)
else:
    # -----------------------------------------------------------
    # RANDOM ROUGH GRID FOR PARAMETER TUNING
    # -----------------------------------------------------------
    patch_size_list, num_epochs_list, batch_size_list, learning_rate_list, dropout_list = read_tuned_params_from_csv(
        tuned_params_file)

    for i in range(len(patch_size_list)):
        patch_size = patch_size_list[i]
        num_epochs = num_epochs_list[i]
        batch_size = batch_size_list[i]
        lr = learning_rate_list[i]
        dropout = dropout_list[i]

        for patient in patients:
            predict_and_save(patch_size, num_epochs, batch_size, lr, dropout, patient, num_patients_train,
                             num_patients_val, data_dirs, dataset)

print('DONE')
