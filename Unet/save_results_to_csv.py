"""
File name: save_results_to_csv.py
Author: Jana Rieger
Date created: 02/21/2018

This script gets the performance measures from the training data set that was already calculated with the train_unet.py
script and additionally calculates the performance measures on the validation or test set as average over all patients
in the particular set. Then it saves all the values for each model to csv file.
"""

import time
import csv
from Unet import config

################################################
# SET PARAMETERS
################################################
from Unet.utils.helper import read_tuned_params_from_csv
from Unet.utils.measure_performance_function import measure_performance_and_save_to_csv

dataset = 'val'
patch_size_list = [96]  # size of one patch n x n
batch_size_list = [16, 32]  # list with batch sizes
num_epochs = 10  # number of epochs
learning_rate_list = [1e-3, 1e-4, 1e-5]  # list with learning rates of the optimizer Adam
dropout_list = [0.2, 0.3, 0.4]  # percentage of weights to be dropped
threshold = config.THRESHOLD
fine_grid = False  # True for tuning hyperparameters in fine grid tuning, False for random rough grid
################################################

num_patients_train = len(config.PATIENTS['train']['working']) + len(
    config.PATIENTS['train']['working_augmented'])  # number of patients in training category
num_patients_val = len(config.PATIENTS['val']['working']) + len(
    config.PATIENTS['val']['working_augmented'])  # number of patients in validation category
patients_segmentation = config.PATIENTS[dataset]['working'] + config.PATIENTS[dataset]['working_augmented']

data_dirs = config.MODEL_DATA_DIRS
tuned_params_file = config.get_tuned_parameters()
result_file = config.get_complete_result_table_filepath(dataset)

header_row = ['patch size', 'num epochs', 'batch size', 'learning rate', 'dropout', 'tp train', 'fn train', 'fp train',
              'tn train', 'auc train', 'acc train', 'avg acc train', 'dice train',
              'tp patch val', 'fn patch val', 'fp patch val',
              'tn patch val', 'auc patch val', 'acc patch val', 'avg acc patch val',
              'dice patch val',
              'tp ' + dataset, 'fn ' + dataset, 'fp ' + dataset,
              'tn ' + dataset, 'auc ' + dataset, 'acc ' + dataset, 'avg acc ' + dataset, 'dice ' + dataset]
with open(result_file, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(header_row)

start_total = time.time()

# PARAMETER LOOPS
if fine_grid:
    # -----------------------------------------------------------
    # FINE GRID FOR PARAMETER TUNING
    # -----------------------------------------------------------
    for patch_size in patch_size_list:
        for batch_size in batch_size_list:
            for lr in learning_rate_list:
                for dropout in dropout_list:
                    measure_performance_and_save_to_csv(patch_size, num_epochs, batch_size, lr, dropout, threshold,
                                                        num_patients_train,
                                                        num_patients_val, patients_segmentation, data_dirs, dataset,
                                                        result_file)
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

        measure_performance_and_save_to_csv(patch_size, num_epochs, batch_size, lr, dropout, threshold,
                                            num_patients_train,
                                            num_patients_val, patients_segmentation, data_dirs, dataset,
                                            result_file)

duration_total = int(time.time() - start_total)
print('total performance assessment took:', (duration_total // 3600) % 60, 'hours', (duration_total // 60) % 60,
      'minutes', duration_total % 60, 'seconds')
print('DONE')
