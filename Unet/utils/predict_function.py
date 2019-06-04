"""
File name: predict_function.py
Author: Jana Rieger
Date created: 03/03/2018

This script contains the function that divides the input MRA image in small patches, predicts the patches with given
model and saves the probability matrix as nifti file.
"""

import pickle
import time
import numpy as np
from keras.models import load_model
from Unet import config
from Unet.utils import helper
from Unet.utils.metrics import dice_coef_loss, dice_coef
import os

if config.TOP_LEVEL == './':
    os.chdir('../')


def predict_and_save(patch_size, num_epochs, batch_size, lr, dropout, patient, num_patients_train, num_patients_val,
                     data_dirs, dataset):
    print('________________________________________________________________________________')
    print('patch size', patch_size)
    print('batch size', batch_size)
    print('learning rate', lr)
    print('dropout', dropout)
    print('patient:', patient)

    # create the name of current run
    run_name = config.get_run_name(patch_size, num_epochs, batch_size, lr, dropout, num_patients_train,
                                   num_patients_val)
    print(run_name)

    # -----------------------------------------------------------
    # LOADING MODEL, RESULTS AND WHOLE BRAIN MATRICES
    # -----------------------------------------------------------
    model_filepath = config.get_model_filepath(run_name)
    print(model_filepath)
    model = load_model(model_filepath,
                       custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

    train_metadata_filepath = config.get_train_metadata_filepath(run_name)
    with open(train_metadata_filepath, 'rb') as handle:
        train_metadata = pickle.load(handle)
    print('train setting: ', train_metadata['params'])

    print('> Loading image...')
    img_mat = helper.load_nifti_mat_from_file(
        data_dirs[dataset] + patient + '_img.nii')  # values between 0 and 255
    print('> Loading mask...')
    mask_mat = helper.load_nifti_mat_from_file(
        data_dirs[dataset] + patient + '_mask.nii')  # values 0 and 1

    # -----------------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------------
    # the segmentation is going to be saved in this probability matrix
    prob_mat = np.zeros(img_mat.shape, dtype=np.float32)
    x_dim, y_dim, z_dim = prob_mat.shape

    # get the x, y and z coordinates where there is brain
    x, y, z = np.where(mask_mat)
    print('x shape:', x.shape)
    print('y shape:', y.shape)
    print('z shape:', z.shape)

    # get the z slices with brain
    z_slices = np.unique(z)

    # start cutting out and predicting the patches
    starttime_total = time.time()
    # proceed slice by slice
    for i in z_slices:
        print('Slice:', i)
        starttime_slice = time.time()
        slice_vox_inds = np.where(z == i)
        # find all x and y coordinates with brain in given slice
        x_in_slice = x[slice_vox_inds]
        y_in_slice = y[slice_vox_inds]
        # find min and max x and y coordinates
        slice_x_min = min(x_in_slice)
        slice_x_max = max(x_in_slice)
        slice_y_min = min(y_in_slice)
        slice_y_max = max(y_in_slice)

        # calculate number of predicted patches in x and y direction in given slice
        num_of_x_patches = np.int(np.ceil((slice_x_max - slice_x_min) / patch_size))
        num_of_y_patches = np.int(np.ceil((slice_y_max - slice_y_min) / patch_size))
        print('num x patches', num_of_x_patches)
        print('num y patches', num_of_y_patches)

        # predict patch by patch in given slice
        for j in range(num_of_x_patches):
            for k in range(num_of_y_patches):
                # find the starting and ending x and y coordinates of given patch
                patch_start_x = slice_x_min + patch_size * j
                patch_end_x = slice_x_min + patch_size * (j + 1)
                patch_start_y = slice_y_min + patch_size * k
                patch_end_y = slice_y_min + patch_size * (k + 1)
                # if the dimensions of the probability matrix are exceeded shift back the last patch
                if patch_end_x > x_dim:
                    patch_end_x = slice_x_max
                    patch_start_x = slice_x_max - patch_size
                if patch_end_y > y_dim:
                    patch_end_y = slice_y_max
                    patch_start_y = slice_y_max - patch_size

                # get the patch with the found coordinates from the image matrix
                img_patch = img_mat[patch_start_x: patch_end_x, patch_start_y: patch_end_y, i]

                # normalize the patch with mean and standard deviation calculated over training set
                img_patch = img_patch.astype(np.float)
                img_patch -= train_metadata['params']['mean']
                img_patch /= train_metadata['params']['std']

                # predict the patch with the model and save to probability matrix
                prob_mat[patch_start_x: patch_end_x, patch_start_y: patch_end_y, i] = np.reshape(
                    model.predict(
                        np.reshape(img_patch,
                                   (1, patch_size, patch_size, 1)), batch_size=1, verbose=0),
                    (patch_size, patch_size))

        # how long does the prediction take for one slice
        duration_slice = time.time() - starttime_slice
        print('prediction in slice took:', (duration_slice // 3600) % 60, 'hours',
              (duration_slice // 60) % 60, 'minutes',
              duration_slice % 60, 'seconds')

    # how long does the prediction take for a patient
    duration_total = time.time() - starttime_total
    print('prediction in total took:', (duration_total // 3600) % 60, 'hours',
          (duration_total // 60) % 60, 'minutes',
          duration_total % 60, 'seconds')

    # -----------------------------------------------------------
    # SAVE AS NIFTI
    # -----------------------------------------------------------
    helper.create_and_save_nifti(prob_mat, config.get_probs_filepath(run_name, patient, dataset))
