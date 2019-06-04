"""
File name: check_augmented_data.py
Author: Jana Rieger
Date created: 02/19/2018

This script plots examples of processed and augmented data.
"""

import nibabel as nib
import matplotlib.pyplot as plt
from Unet.utils import helper

nr_of_rotations_for_plotting = -1

slice = 50

# Let's load it and look at it:
img_0005 = nib.load('../../model_data/train/0005_img.nii').get_data()
label_0005 = nib.load('../../model_data/train/0005_label.nii').get_data()
img_1005 = nib.load('../../model_data/train/1005_img.nii').get_data()
label_1005 = nib.load('../../model_data/train/1005_label.nii').get_data()
helper.plot_some_images([img_0005, label_0005, img_1005, label_1005], 1, 80, 100, '0005 & 1005',
                        nr_of_rotations_for_plotting, slice)

img_0031 = nib.load('../../model_data/val/0031_img.nii').get_data()
label_0031 = nib.load('../../model_data/val/0031_label.nii').get_data()
img_1031 = nib.load('../../model_data/val/1031_img.nii').get_data()
label_1031 = nib.load('../../model_data/val/1031_label.nii').get_data()
helper.plot_some_images([img_0031, label_0031, img_1031, label_1031], 2, 680, 100, '0031 & 1031',
                        nr_of_rotations_for_plotting, slice)

img_0035 = nib.load('../../model_data/test/0035_img.nii').get_data()
label_0035 = nib.load('../../model_data/test/0035_label.nii').get_data()
img_1035 = nib.load('../../model_data/test/1035_img.nii').get_data()
label_1035 = nib.load('../../model_data/test/1035_label.nii').get_data()
helper.plot_some_images([img_0035, label_0035, img_1035, label_1035], 3, 1270, 100, '0035 & 1035',
                        nr_of_rotations_for_plotting, slice)

plt.show()
