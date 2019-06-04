"""
File name: evaluate_segmentation.py
Author: Jana Rieger
Date created: 03/25/2018

This script uses the Unet/Evaluate_segmentation.exe (Abdel Aziz Taha and Allan Hanbury. Metrics for evaluating
3D medical image segmentation: analysis, selection, and tool. BMC Medical Imaging, 15:29, August 2015.
https://github.com/Visceral-Project/EvaluateSegmentation) for calculating performance measures.
"""

import time
from Unet import config
import os
from Unet.utils.evaluate_segmentation_functions import evaluate_segmentation
from Unet.utils.helper import read_tuned_params_from_csv

################################################
# SET PARAMETERS
################################################
dataset = 'test'
patch_size_list = [96]  # size of one patch n x n
batch_size_list = [8]  # list with batch sizes
num_epochs = 10  # number of epochs
learning_rate_list = [1e-5]  # list with learning rates of the optimizer Adam
dropout_list = [0.0]  # percentage of weights to be dropped
threshold = config.THRESHOLD
fine_grid = True  # True for tuning hyperparameters in fine grid tuning, False for random rough grid
measures = "DICE,JACRD,AUC,KAPPA,RNDIND,ADJRIND,ICCORR,VOLSMTY,MUTINF,HDRFDST@0.95@,AVGDIST,MAHLNBS,VARINFO,GCOERR," \
           "PROBDST,SNSVTY,SPCFTY,PRCISON,FMEASR,ACURCY,FALLOUT,TP,FP,TN,FN,REFVOL,SEGVOL"
################################################

num_patients_train = len(config.PATIENTS['train']['working']) + len(
    config.PATIENTS['train']['working_augmented'])  # number of patients in training set
num_patients_val = len(config.PATIENTS['val']['working']) + len(
    config.PATIENTS['val']['working_augmented'])  # number of patients in validation set
patients_segmentation = config.PATIENTS[dataset]['working'] + config.PATIENTS[dataset]['working_augmented']
data_dirs = config.MODEL_DATA_DIRS

# create results folder for evaluation segmentation
results_path = config.EVAL_SEGMENT_DIR
if not os.path.exists(results_path):
    os.makedirs(results_path)
executable_path = config.EXECUTABLE_PATH
csv_path = config.get_eval_segment_dataset_csvpath(dataset)
csv_path_per_patient = config.get_eval_segment_dataset_csvpath_per_patient(dataset)
tuned_params_file = config.get_tuned_parameters()

# PARAMETER LOOPS
start_total = time.time()
if fine_grid:
    # -----------------------------------------------------------
    # FINE GRID FOR PARAMETER TUNING
    # -----------------------------------------------------------
    for patch_size in patch_size_list:
        for batch_size in batch_size_list:
            for lr in learning_rate_list:
                for dropout in dropout_list:
                    evaluate_segmentation(patch_size, num_epochs, batch_size, lr, dropout, num_patients_train,
                                          num_patients_val,
                                          patients_segmentation, threshold, data_dirs, dataset, executable_path,
                                          csv_path_per_patient,
                                          csv_path, measures)
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

        evaluate_segmentation(patch_size, num_epochs, batch_size, lr, dropout, num_patients_train,
                              num_patients_val,
                              patients_segmentation, threshold, data_dirs, dataset, executable_path,
                              csv_path_per_patient,
                              csv_path, measures)

duration_total = int(time.time() - start_total)
print('performance assessment took:', (duration_total // 3600) % 60, 'hours', (duration_total // 60) % 60, 'minutes',
      duration_total % 60,
      'seconds')
print('DONE')
