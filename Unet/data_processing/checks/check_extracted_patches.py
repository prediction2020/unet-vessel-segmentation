"""
File name: check_extracted_patches.py
Author: Jana Rieger
Date created: 02/26/2018

This scripts checks if the correct number of patches was extracted, calculates the number of patches containing no
vessels and plots samples of extracted patches.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from Unet import config

# change current working directory to top-level module (Unet)
os.chdir('../../')
print(os.getcwd())

# directories, paths and filenames
data_dir = config.MODEL_DATA_DIRS
patient_files = config.PATIENTS

nr_rotations_for_plotting = -1
plot_patches = False  # if to plot extracted patches

patch_sizes = config.PATCH_SIZES  # different quadratic patch sizes n x n
nr_patches = config.NUM_PATCHES  # number of patches we want to extract from one stack (one patient)
nr_patches_to_plot = 20
start = 495

num_empty_labels_total = {}
num_patients = 0
for size in patch_sizes:
    num_empty_labels_total[size] = 0
for category, patients in patient_files.items():
    if category == 'test':
        break

    all_patients = patients['working'] + patients['working_augmented']
    for p, patient in enumerate(all_patients):
        num_patients += 1
        imgs = {}
        labels = {}
        for size in patch_sizes:
            print('DATA CATEGORY:', category)
            print('PATIENT', patient)
            print('SIZE:', size)

            patch_path = data_dir[category] + 'patch' + str(size) + '/' + str(patient) + '_' + str(size)

            imgs[size] = np.load(patch_path + '_img.npy')
            print('Correct number of patches in imgs:', len(imgs[size]) == nr_patches)
            assert len(imgs[size]) == nr_patches, 'Number of patches in images is' + str(
                len(imgs[size])) + ' but should be ' + str(nr_patches)

            labels[size] = np.load(patch_path + '_label.npy')
            print('Correct number of patches in labels:', len(labels[size]) == nr_patches)
            assert len(labels[size]) == nr_patches, 'Number of patches in images is' + str(
                len(labels[size])) + ' but should be ' + str(nr_patches)

            num_empty_labels = 0
            for label in labels[size]:
                if label.sum() == 0:
                    num_empty_labels += 1
            percentage_empty_labels = num_empty_labels / nr_patches * 100
            num_empty_labels_total[size] += num_empty_labels
            print('Percentage of empty labels:', percentage_empty_labels, '%')

        # visualize some patches
        if plot_patches:
            print('Plotting...')
            fig = plt.figure(p + 1, figsize=(10, 9))
            plt.suptitle(category + " " + patient)
            fig.subplots_adjust(top=0.90, bottom=0.01)
            num_columns = len(patch_sizes) * 2
            for i in range(nr_patches_to_plot):
                for j, size in enumerate(patch_sizes):
                    plt.subplot(nr_patches_to_plot, num_columns, (i * num_columns) + (j * 2 + 1))
                    plt.title(size) if i == 0 else 0
                    plt.axis('off')
                    plt.imshow(imgs[size][start + i])

                    plt.subplot(nr_patches_to_plot, num_columns, (i * num_columns) + (j * 2 + 2))
                    plt.axis('off')
                    plt.imshow(labels[size][start + i])

                    j += 1

        plt.show()
print()
print('Percentage of empty labels in total:')
for size in patch_sizes:
    print('Patch size', size, ':', num_empty_labels_total[size] / (nr_patches * num_patients) * 100, '%')
print('DONE')
