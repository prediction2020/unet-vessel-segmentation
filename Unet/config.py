"""
File name: config.py
Author: Jana Rieger
Date created: 02/10/2018

This file contains constants like paths, names, sizes and general parameters of the model which are used in other
scripts.
"""

import os
import time
from Unet.utils.metrics import dice_coef_loss, dice_coef
from keras.optimizers import Adam

print(os.getcwd())

# comment and uncomment following 2 lines according to where you run the program
TOP_LEVEL = '.'  # for running in pycharm
# TOP_LEVEL = 'Unet'  # for running in terminal in the segmentation directory with command: python -m Unet.train_unet

# -----------------------------------------------------------
# PATCH SETTINGS FOR EXTRACTION
# -----------------------------------------------------------
PATCH_SIZES = [16, 32, 64, 96]  # different quadratic patch sizes n x n
NUM_PATCHES = 1000  # number of patches we want to extract from one stack (one patient)

# -----------------------------------------------------------
# GENERAL MODEL PARAMETERS
# -----------------------------------------------------------
NUM_CHANNELS = 1  # number of channels of the input images
ACTIVATION = 'relu'  # activation_function after every convolution
FINAL_ACTIVATION = 'sigmoid'  # activation_function of the final layer
LOSS_FUNCTION = dice_coef_loss  # dice loss function defined in Unet/utils/metrics file # 'binary_crossentropy'
METRICS = [dice_coef, 'accuracy']  # , avg_class_acc # dice coefficient defined in Unet/utils/metrics file
OPTIMIZER = Adam  # Adam: algorithm for first-order gradient-based optimization of stochastic objective functions
THRESHOLD = 0.5  # threshold for getting classes from probability maps

# -----------------------------------------------------------
# DIRECTORIES, PATHS AND FILE NAMES
# -----------------------------------------------------------
# directory where the original scans are stored
ORIGINAL_DATA_DIR = TOP_LEVEL + '/original_data/'
# original files with scans
IMG_FILENAME = '001_ODCT_Ric_denoised_NUC.nii'
MASK_FILENAME = '001_NUC_BET_mask.nii'
LABEL_FILENAME = '001_Vessel_Manual_Gold_int.nii'
# directories where masked and augmented data are stored
MODEL_DATA_DIRS = {'train': TOP_LEVEL + '/model_data/train/',
                   'test': TOP_LEVEL + '/model_data/test/',
                   'val': TOP_LEVEL + '/model_data/val/'}
# matching the patient to train / val / test set, conversion from original patient names to working names
# according to conversion table:
# https://docs.google.com/spreadsheets/d/1Q7RaHS03XAVtqVFpReuokHZHsu_n3pmr8xNi9a6yeVw/edit#gid=0
PATIENTS = {
    'train': {
        'original': [
            'PEG0005', 'PEG0006', 'PEG0008',
            'PEG0009', 'PEG0010', 'PEG0011', 'PEG0012', 'PEG0013', 'PEG0014', 'PEG0015', 'PEG0016',
            'PEG0017', 'PEG0018', 'PEG0019', 'PEG0020', 'PEG0021', 'PEG0022', 'PEG0023',
            'PEG0036', 'PEG0037', 'PEG0038', 'PEG0039', 'PEG0040', 'PEG0041', 'PEG0042', 'PEG0043', 'PEG0044',
            'PEG0045', 'PEG0046', 'PEG0047', 'PEG0048', 'PEG0058', 'PEG0059', 'PEG0060', 'PEG0061', 'PEG0062',
            'PEG0063', 'PEG0064', 'PEG0066', 'PEG0068', 'PEG0069'
        ],
        'working': [
            '0005', '0006', '0008',
            '0009', '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020', '0021',
            '0022', '0023',
            '0036', '0037', '0038', '0039', '0040', '0041', '0042', '0043', '0044', '0045', '0046', '0047', '0048',
            '0058', '0059', '0060', '0061', '0062', '0063', '0064', '0066', '0068', '0069'
        ],
        'working_augmented': [
            # '1005', '1006', '1008' # example from previous version of the framework, augmentation in this version is
            # done on-the-fly with ImageDataGenerator from Keras, these 'working_augmented' lists must stay empty!
        ]
    },
    'val': {
        'original': [
            'PEG0024', 'PEG0025', 'PEG0026', 'PEG0027', 'PEG0028',
            'PEG0070', 'PEG0071', 'PEG0072', 'PEG0073', 'PEG0074', 'PEG0075'
        ],
        'working': [
            '0024', '0025', '0026', '0027', '0028',
            '0070', '0071', '0072', '0073', '0074', '0075'
        ],
        'working_augmented': []
    },  # in this version these 'working_augmented' lists must stay empty!
    'test': {
        'original': [
            'PEG0029', 'PEG0030', 'PEG0031', 'PEG0032', 'PEG0033', 'PEG0034', 'PEG0035',
            'PEG0076', 'PEG0077', 'PEG0078', 'PEG0079', 'PEG0080', 'PEG0081', 'PEG0082'
        ],
        'working': [
            '0029', '0030', '0031', '0032', '0033', '0034', '0035',
            '0076', '0077', '0078', '0079', '0080', '0081', '0082'
        ],
        'working_augmented': []  # in this version these 'working_augmented' lists must stay empty!
    }
}

# directories for saving models and results
MODELS_DIR = TOP_LEVEL + '/models/'
RESULTS_DIR = TOP_LEVEL + '/results/'
IMGS_DIR = TOP_LEVEL + '/imgs/'


# title for saving the results of one training run
def get_run_name(patch_size, num_epochs, batch_size, learning_rate, dropout, num_patients_train, num_patients_val):
    return '_patchsize_' + str(patch_size) + '_epochs_' + str(num_epochs) + '_batchsize_' + str(
        batch_size) + '_lr_' + str(learning_rate) + '_dropout_' + str(dropout) + '_train_' + str(
        num_patients_train) + '_val_' + str(num_patients_val)


# where to store the trained model
def get_model_filepath(run_name):
    return MODELS_DIR + 'model' + run_name + '.h5py'


# where to store the results of training with parameters and training history
def get_train_metadata_filepath(run_name):
    return MODELS_DIR + 'train_metadata' + run_name + '.pkl'


# where to store csv file with training history
def get_train_history_filepath(run_name):
    return MODELS_DIR + 'train_history' + run_name + '.csv'


# where to save probability map from validation as nifti
def get_probs_filepath(run_name, patient, category):
    return RESULTS_DIR + category + '/' + category + '_probs_' + patient + '_' + run_name + '.nii'


def get_tuned_parameters():
    return MODELS_DIR + 'tuned_params.csv'


# where to save complete result table as csv
def get_complete_result_table_filepath(category):
    return RESULTS_DIR + 'result_table_' + category + '_' + time.strftime("%Y%m%d-%H%M%S") + '.csv'


# -----------------------------------------------------------
# EVALUATE SEGMENTATION TOOL
# -----------------------------------------------------------
EXECUTABLE_PATH = 'EvaluateSegmentation.exe'
EVAL_SEGMENT_DIR = RESULTS_DIR + 'eval_segment/'
TEMP_LABEL_PATH = EVAL_SEGMENT_DIR + 'temp_label.nii'
TEMP_SEGMENTATION_PATH = EVAL_SEGMENT_DIR + 'temp_segmentation.nii'


# where to save evaluate segmentation result for all patient in category (val / test) as xml
def get_eval_segment_dataset_xmlpath(run_name, patient, dataset):
    return EVAL_SEGMENT_DIR + 'eval_segment_' + dataset + '_' + patient + run_name + '.xml'


# where to save evaluate segmentation result for all patient in category (val / test) as csv
def get_eval_segment_dataset_csvpath_per_patient(dataset):
    return EVAL_SEGMENT_DIR + 'eval_segment_' + dataset + '_per_patient_' + time.strftime("%Y%m%d-%H%M%S") + '.csv'


# where to save evaluate segmentation result for all patient in category (val / test) as csv
def get_eval_segment_dataset_csvpath(dataset):
    return EVAL_SEGMENT_DIR + 'eval_segment_' + dataset + '_' + time.strftime("%Y%m%d-%H%M%S") + '.csv'
