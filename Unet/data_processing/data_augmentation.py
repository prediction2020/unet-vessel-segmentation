"""
File name: data_augmentation.py
Author: Jana Rieger
Date created: 02/10/2018

This script loads the original scans, masks and ground truth labels. Then it applies the masks to the original images
and saves the skull stripped images under a new name defined in config.py ('working') into 'Unet/model_data/' directory.
Then the images in the train set are augmented by flipping over the x-axis and saved under the names for augmented
versions defined in config.py ('working_augmented') into 'Unet/model_data/' directory.
"""

import os
import matplotlib.pyplot as plt
from Unet import config
from Unet.utils import helper

# change current working directory to top-level module (Unet)
os.chdir('../')
print(os.getcwd())

# directories, paths and filenames
original_data_dir = config.ORIGINAL_DATA_DIR
patient_files = config.PATIENTS
img_filename = config.IMG_FILENAME
mask_filename = config.MASK_FILENAME
label_filename = config.LABEL_FILENAME
target_dirs = config.MODEL_DATA_DIRS

nr_rotations_for_plotting = -1  # direction and number of rot90 just for plotting
plot = False  # if to plot some images for visualising of the loaded and processed data

slice_dimensions = (312, 384)

# process data in every set (train/test/val) and every patient in those sets (PEG0005, PEG0006, ...)
for dataset, patients in patient_files.items():
    num_patients = len(patients['original'])

    # the number of patient must be same in every category (original, working, working augmented) defined in the patient
    # dictionary in the config.py
    if len(patients['working_augmented']) != 0:
        assert num_patients == len(patients['working']) == len(
            patients[
                'working_augmented']), "Check config.PATIENT_FILES. The lists with names do not have the same length."
    else:
        assert num_patients == len(
            patients['working']), "Check config.PATIENT_FILES. The lists with names do not have the same length."

    for i in range(num_patients):
        print('DATA SET: ', dataset)
        if len(patients['working_augmented']) != 0:
            print(patients['original'][i], patients['working'][i], patients['working_augmented'][i])
        else:
            print(patients['original'][i], patients['working'][i])

        # load image, mask and label stacks as matrices
        print('Loading image...')
        img_mat = helper.load_nifti_mat_from_file(
            original_data_dir + patients['original'][i] + '/' + img_filename)
        print('Loading mask...')
        mask_mat = helper.load_nifti_mat_from_file(
            original_data_dir + patients['original'][i] + '/' + mask_filename)
        print('Loading label...')
        label_mat = helper.load_nifti_mat_from_file(
            original_data_dir + patients['original'][i] + '/' + label_filename)
        if plot:  # visualize some slices
            helper.plot_some_images([img_mat, mask_mat, label_mat], 1, 80, 100, patients['original'][i],
                                    nr_rotations_for_plotting, 60)

        # check the dimensions
        assert img_mat.shape == mask_mat.shape == label_mat.shape, "The DIMENSIONS of image, mask and label are NOT " \
                                                                   "SAME."
        assert helper.correct_dimensions([img_mat, mask_mat, label_mat],
                                         slice_dimensions), "The DIMENSIONS of image, mask and label are NOT (312, " \
                                                            "384)."

        # mask images and labels (skull stripping)
        img_mat = helper.aplly_mask(img_mat, mask_mat)
        label_mat = helper.aplly_mask(label_mat, mask_mat)
        # save to new file as masked version of original data (PEG0005 -> 0005, PEG0006 -> 0006, ...)
        if not os.path.exists(target_dirs[dataset]):
            os.makedirs(target_dirs[dataset])
        helper.create_and_save_nifti(img_mat, target_dirs[dataset] + patients['working'][i] + '_img.nii')
        helper.create_and_save_nifti(mask_mat, target_dirs[dataset] + patients['working'][i] + '_mask.nii')
        helper.create_and_save_nifti(label_mat, target_dirs[dataset] + patients['working'][i] + '_label.nii')
        if plot:  # visualize some images
            helper.plot_some_images([img_mat, label_mat], 2, 680, 100, patients['working'][i],
                                    nr_rotations_for_plotting, 60)

        # flip to augment the masked images and labels only if set in config.py
        if len(patients['working_augmented']) != 0:
            img_mat = helper.flip(img_mat, axis=0)
            mask_mat = helper.flip(mask_mat, axis=0)
            label_mat = helper.flip(label_mat, axis=0)
            # save to new file as augmented version of masked data (0001 -> 1001, 0002 -> 1002, ...)
            helper.create_and_save_nifti(img_mat, target_dirs[dataset] + patients['working_augmented'][i] + '_img.nii')
            helper.create_and_save_nifti(mask_mat,
                                         target_dirs[dataset] + patients['working_augmented'][i] + '_mask.nii')
            helper.create_and_save_nifti(label_mat,
                                         target_dirs[dataset] + patients['working_augmented'][i] + '_label.nii')
            if plot:  # visualize some images
                helper.plot_some_images([img_mat, label_mat], 3, 1270, 100, patients['working_augmented'][i],
                                        nr_rotations_for_plotting, 60)

        if plot:
            plt.show()
        print()
print('DONE')
