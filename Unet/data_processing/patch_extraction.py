"""
File name: patch_extraction.py
Author: Jana Rieger
Date created: 02/12/2018

This script extracts the patches in different sizes from the images and corresponding ground truth labels
from the train data set.
"""

import os
import numpy as np
from Unet.utils import helper
from Unet import config

# change current working directory to top-level module (Unet)
os.chdir('../')
print(os.getcwd())

# directories, paths and file names
data_dirs = config.MODEL_DATA_DIRS
patient_files = config.PATIENTS
print('Patient files:', patient_files)

patch_sizes = config.PATCH_SIZES  # different quadratic patch sizes n x n
patch_sizes.sort()  # just in case if the list of patch sizes is not in the ascendant order
nr_patches = config.NUM_PATCHES  # number of patches we want to extract from one stack (one patient)
nr_vessel_patches = nr_patches // 2  # patches that are extracted around vessels
nr_empty_patches = nr_patches - nr_vessel_patches  # patches that are extracted from the brain region but not around
# vessels

# extract patches from each data stack (patient)
for dataset, patients in patient_files.items():
    # the number of patient must be same in every category (original, working, working augmented) defined in the patient
    # dictionary in the config.py
    if len(patients['working_augmented']) != 0:
        assert len(patients['original']) == len(patients['working']) == len(
            patients[
                'working_augmented']), "Check config.PATIENT_FILES. The lists with names do not have the same length."
    else:
        assert len(patients['original']) == len(
            patients['working']), "Check config.PATIENT_FILES. The lists with names do not have the same length."

    all_patients = patients['working'] + patients['working_augmented']
    print('All patients:', all_patients)

    for patient in all_patients:
        print('DATA SET: ', dataset)
        print('PATIENT: ', patient)

        # load image and label stacks as matrices
        print('> Loading image...')
        img_mat = helper.load_nifti_mat_from_file(
            data_dirs[dataset] + patient + '_img.nii')  # values between 0 and 255
        print('> Loading mask...')
        mask_mat = helper.load_nifti_mat_from_file(data_dirs[dataset] + patient + '_mask.nii')  # values 0 or 1
        print('> Loading label...')
        label_mat = helper.load_nifti_mat_from_file(data_dirs[dataset] + patient + '_label.nii')  # values 0 or 1

        current_nr_extracted_patches = 0  # counts already extracted patches
        img_patches = {}  # dictionary to save image patches
        label_patches = {}  # dictionary to save label patches
        # make lists in dictionaries for each extracted patch size
        for size in patch_sizes:
            img_patches[str(size)] = []
            label_patches[str(size)] = []

        # variables with sizes and ranges for searchable areas
        max_patch_size = max(patch_sizes)
        half_max_size = max_patch_size // 2
        max_row = label_mat.shape[0] - max_patch_size // 2
        max_col = label_mat.shape[1] - max_patch_size // 2

        # -----------------------------------------------------------
        # EXTRACT RANDOM PATCHES WITH VESSELS IN THE CENTER OF EACH PATCH
        # -----------------------------------------------------------
        # cut off half of the biggest patch on the edges to create the searchable area -> to ensure that there will be
        # enough space for getting the patch
        searchable_label_area = label_mat[half_max_size: max_row, half_max_size: max_col, :]
        # find all vessel voxel indices in searchable area
        vessel_inds = np.asarray(np.where(searchable_label_area == 1))

        # keep extracting patches while the desired number of patches has not been reached yet, this just in case some
        # patches would be skipped, because they were already extracted (we do not want to have same patches in the set
        # more than once)
        while current_nr_extracted_patches < nr_vessel_patches * len(patch_sizes):
            # find given number of random vessel indices
            random_vessel_inds = vessel_inds[:,
                                 np.random.choice(vessel_inds.shape[1], nr_vessel_patches, replace=False)]
            for i in range(nr_vessel_patches):
                # stop extracting if the desired number of patches has been reached
                if current_nr_extracted_patches == nr_vessel_patches * len(patch_sizes):
                    break

                # get the coordinates of the random vessel around which the patch will be extracted
                x = random_vessel_inds[0][i] + half_max_size
                y = random_vessel_inds[1][i] + half_max_size
                z = random_vessel_inds[2][i]

                # extract patches of different quadratic sizes with the random vessel voxel in the center of each patch
                for size in patch_sizes:
                    half_size = size // 2
                    random_img_patch = img_mat[x - half_size:x + half_size, y - half_size:y + half_size, z]
                    random_label_patch = label_mat[x - half_size:x + half_size, y - half_size:y + half_size, z]

                    # just sanity check if the patch is already in the list
                    if any((random_img_patch == x).all() for x in img_patches[str(size)]):
                        print('Skip patch because already extracted. size:', size)
                        break
                    else:
                        # append the extracted patches to the dictionaries
                        img_patches[str(size)].append(random_img_patch)
                        label_patches[str(size)].append(random_label_patch)
                        current_nr_extracted_patches += 1
                        if current_nr_extracted_patches % 100 == 0:
                            print(current_nr_extracted_patches, 'PATCHES CREATED')

        # -----------------------------------------------------------
        # EXTRACT RANDOM EMPTY PATCHES
        # -----------------------------------------------------------
        # cut off half of the biggest patch on the edges to create the searchable area -> to ensure that there will be
        # enough space for getting the patch
        searchable_mask_area = mask_mat[half_max_size: max_row, half_max_size: max_col, :]
        # find all brain voxel indices
        brain_inds = np.asarray(np.where(searchable_mask_area == 1))

        # keep extracting patches while the desired number of patches has not been reached yet, this just in case some
        # patches would be skipped, because they were already extracted (we do not want to have same patches in the set
        # more than once)
        while current_nr_extracted_patches < nr_patches * len(patch_sizes):
            # find given number of random indices in the brain area
            random_brain_inds = brain_inds[:, np.random.choice(brain_inds.shape[1], nr_empty_patches, replace=False)]
            for i in range(nr_empty_patches):
                # stop extracting if the desired number of patches has been reached
                if current_nr_extracted_patches == nr_patches * len(patch_sizes):
                    break

                # get the coordinates of the random brain voxel around which the patch will be extracted
                x = random_brain_inds[0][i] + half_max_size
                y = random_brain_inds[1][i] + half_max_size
                z = random_brain_inds[2][i]

                # extract patches of different quadratic sizes with the random brain voxel in the center of each patch
                for size in patch_sizes:
                    half_size = size // 2
                    random_img_patch = img_mat[x - half_size:x + half_size, y - half_size:y + half_size, z]
                    random_label_patch = label_mat[x - half_size:x + half_size, y - half_size:y + half_size, z]

                    # just sanity check if the patch is already in the list
                    if any((random_img_patch == x).all() for x in img_patches[str(size)]):
                        print('Skip patch because already extracted. size:', size)
                        break
                    else:
                        # append the extracted patches to the dictionaries
                        img_patches[str(size)].append(random_img_patch)
                        label_patches[str(size)].append(random_label_patch)
                        current_nr_extracted_patches += 1
                        if current_nr_extracted_patches % 100 == 0:
                            print(current_nr_extracted_patches, 'PATCHES CREATED')

        assert current_nr_extracted_patches == nr_patches * len(
            patch_sizes), "The number of extracted patches is  " + str(
            current_nr_extracted_patches) + " but should be " + str(
            nr_patches * len(patch_sizes))

        # save extracted patches as numpy arrays
        for size in patch_sizes:
            print('number of extracted image patches:', len(img_patches[str(size)]))
            print('number of extracted label patches:', len(label_patches[str(size)]))
            directory = data_dirs[dataset] + 'patch' + str(size) + '/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            np.save(directory + patient + '_' + str(size) + '_img', np.asarray(img_patches[str(size)]))
            np.save(directory + patient + '_' + str(size) + '_label', np.asarray(label_patches[str(size)]))
            print('Image patches saved to', directory + patient + '_' + str(size) + '_img.npy')
            print('Label patches saved to', directory + patient + '_' + str(size) + '_label.npy')
        print()
print('DONE')
