"""
File name: prepare_train_val_sets.py
Author: Jana Rieger
Date created: 02/24/2018

This helper script loads the patches for every patient in the data sets train and val and stacks them to one matrix
so that they can be fed into the model for training.
"""

import numpy as np
from Unet import config
from Unet.utils import helper


def get_feature_label_set(dataset, patch_size, num_patches, patients):
    """
    Loads the patches for every patient in the given data set and stacks them to one matrix.

    :param dataset: String, train or val.
    :param patch_size: Number, for what patch size the data sets shall be created.
    :param num_patches: Number, how many patches of one size were extracted from one patient.
    :param patients: Dictionary with patient names, same structure as defined in config.py.
    :return: X - features, y - labels.
    """
    # the number of patient must be same in every category (original, working, working augmented) defined in the patient
    # dictionary in the config.py
    if len(patients[dataset]['working_augmented']) != 0:
        assert len(patients[dataset]['original']) == len(patients[dataset]['working']) == len(
            patients[dataset][
                'working_augmented']), "Check config.PATIENT_FILES. The lists with names do not have the same length."
    else:
        assert len(patients[dataset]['original']) == len(patients[dataset][
                                                                  'working']), \
            "Check config.PATIENT_FILES. The lists with names do not have the same length."

    # get all patients (working + working augmented)
    all_patients_in_dataset = patients[dataset]['working'] + patients[dataset]['working_augmented']
    num_patients_in_dataset = len(all_patients_in_dataset)
    print('number of patients:', num_patients_in_dataset)

    # prepare empty matrices for features (X) and labels (y) to store the loaded patches for every patient
    X = np.ndarray(
        (num_patients_in_dataset * num_patches, patch_size, patch_size))
    y = np.ndarray(
        (num_patients_in_dataset * num_patches, patch_size, patch_size), dtype=np.uint8)

    # load patches from each patient and save to the prepared empty matrix
    for i, patient in enumerate(all_patients_in_dataset):
        imgs, labels = helper.load_patches(config.MODEL_DATA_DIRS[dataset], patient, patch_size)
        X[i * num_patches: (i + 1) * num_patches, :, :] = imgs
        y[i * num_patches: (i + 1) * num_patches, :, :] = labels

    # reshape to (nr of samples, patch height, patch width, nr of channels)
    X = X.reshape(X.shape[0], X.shape[2], X.shape[2], 1)
    y = y.reshape(y.shape[0], y.shape[2], y.shape[2], 1)
    print(dataset, 'X', X.shape)
    print(dataset, 'y', y.shape)

    return X, y


def create_training_datasets(patch_size, num_patches, patients):
    """
    Gets the patches for every patient in the data sets train and val for given patch size as a matrix. Normalize them
    to have zero mean and unit variance. And returns them as well as the respective mean and standard deviation.

    :param patch_size: Number, for what patch size the data sets shall be created.
    :param num_patches: Number, how many patches of one size were extracted from one patient.
    :param patients: Dictionary with patient names, same structure as defined in config.py.
    :return: Train feature set, Train label set, Validation feature set, Validation label set, Respective mean,
    Respective standard deviation.
    """
    train_X, train_y = get_feature_label_set('train', patch_size, num_patches, patients)
    val_X, val_y = get_feature_label_set('val', patch_size, num_patches, patients)

    # normalize data: mean and std calculated on the train set and applied to the train and val set
    mean = np.mean(train_X)  # mean for data centering
    std = np.std(train_X)  # std for data normalization
    train_X -= mean
    train_X /= std
    val_X -= mean
    val_X /= std

    return train_X, train_y, val_X, val_y, mean, std
