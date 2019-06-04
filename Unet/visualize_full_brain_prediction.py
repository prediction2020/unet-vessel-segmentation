"""
File name: visualize_full_brain_prediction.py
Author: Jana Rieger
Date created: 03/21/2018

This script plots the predicted segmentation together with the input image, ground-truth label and error map.
There are two plots generated. 1. four different slices of the matrices. 2. scrollable visualisation of the whole
matrices.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from Unet import config
from Unet.utils import helper
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from Unet.utils.metrics import avg_class_acc

################################################
# SET PARAMETERS
################################################
dataset = 'test'
patient = '0029'
patch_size = 16
num_epochs = 10
batch_size = 64
learning_rate = 1e-4
dropout = 0.0
threshold = config.THRESHOLD
num_patients_train = len(config.PATIENTS['train']['working']) + len(
    config.PATIENTS['train']['working_augmented'])  # number of patients in training category
num_patients_val = len(config.PATIENTS['val']['working']) + len(
    config.PATIENTS['val']['working_augmented'])  # number of patients in validation category
################################################

data_dirs = config.MODEL_DATA_DIRS
run_name = config.get_run_name(patch_size, num_epochs, batch_size, learning_rate, dropout, num_patients_train,
                               num_patients_val)
print(run_name)
print('Patient:', patient)
print()

# -----------------------------------------------------------
# LOADING RESULTS
# -----------------------------------------------------------
print('> Loading image ...')
img_mat = helper.load_nifti_mat_from_file(data_dirs[dataset] + patient + '_img.nii')
print('> Loading label...')
label_mat = helper.load_nifti_mat_from_file(data_dirs[dataset] + patient + '_label.nii')
print('> Loading probability map...')
prob_mat = helper.load_nifti_mat_from_file(config.get_probs_filepath(run_name, patient, dataset))
pred_class = (prob_mat > threshold).astype(np.uint8)  # convert from boolean to int

converted_pred_class = pred_class.copy().astype(int)
converted_pred_class[converted_pred_class == 1] = -1
converted_pred_class[converted_pred_class == 0] = 1
error_map = label_mat - converted_pred_class

print()
print("Performance:")
label_mat_f = label_mat.flatten()
prob_mat_f = prob_mat.flatten()
pred_class_f = pred_class.flatten().astype(np.uint8)

pat_auc = roc_auc_score(label_mat_f, prob_mat_f)
pat_acc = accuracy_score(label_mat_f, pred_class_f)
pat_avg_acc, tn, fp, fn, tp = avg_class_acc(label_mat_f, pred_class_f)
pat_dice = f1_score(label_mat_f, pred_class_f)
print('auc:', pat_auc)
print('acc:', pat_acc)
print('avg acc:', pat_avg_acc)
print('dice:', pat_dice)
print()

# -----------------------------------------------------------
# VISUALIZATION OF PICKED SLICES
# -----------------------------------------------------------
print('Plotting...')
fig = plt.figure(1, figsize=(9, 9.5))

slices = [37, 65, 77, 105]
num_columns = 5
cmap1 = 'gray'
cmap2 = 'gist_heat'
cmap3 = ListedColormap(['black', 'cyan', 'red', 'white'])
cmap3.set_over('1')
cmap3.set_under('0')
fontsize=12

for i, sl in enumerate(slices):
    plt.subplot(len(slices), num_columns, (i * num_columns) + 1)
    plt.title('MRA scan', fontsize=fontsize) if i == 0 else 0
    plt.ylabel('Slice ' + str(sl), rotation='vertical', labelpad=4, fontsize=fontsize)
    plt.yticks([])
    plt.xticks([])
    plt.imshow(img_mat.T[sl], cmap=cmap1)

    plt.subplot(len(slices), num_columns, (i * num_columns) + 2)
    plt.title('Predicted \n probability map', fontsize=fontsize) if i == 0 else 0
    plt.yticks([])
    plt.xticks([])
    plt.imshow(prob_mat.T[sl], cmap=cmap2)

    cax = plt.axes([0.23, 0.04, 0.188, 0.014])  # [0.39, 0.0265, 0.007, 0.893]
    plt.colorbar(cax=cax, orientation='horizontal')

    plt.subplot(len(slices), num_columns, (i * num_columns) + 3)
    plt.title('Predicted \n segmentation', fontsize=fontsize) if i == 0 else 0
    plt.yticks([])
    plt.xticks([])
    plt.imshow(pred_class.T[sl], cmap=cmap1)

    plt.subplot(len(slices), num_columns, (i * num_columns) + 4)
    plt.title('Ground-truth', fontsize=fontsize) if i == 0 else 0
    plt.yticks([])
    plt.xticks([])
    plt.imshow(label_mat.T[sl], cmap=cmap1)

    plt.subplot(len(slices), num_columns, (i * num_columns) + 5)
    plt.title('Error map', fontsize=fontsize) if i == 0 else 0
    plt.yticks([])
    plt.xticks([])
    plt.imshow(error_map.T[sl], cmap=cmap3)

    cax2 = plt.axes([0.8021, 0.04, 0.188, 0.014])
    cbar2 = plt.colorbar(cax=cax2, orientation='horizontal', ticks=[-0.6, 0.1, 0.85, 1.6])
    cbar2.ax.set_xticklabels(['TN', 'FN', 'FP', 'TP'])

fig.subplots_adjust(left=0.04, bottom=0.06, right=0.99, top=0.935,
                    wspace=0.02, hspace=0.015)

# plt.show()


# -----------------------------------------------------------
# SCROLLABLE VISUALIZATION OF WHOLE BRAIN
# -----------------------------------------------------------


def scrollable_slice_viewer(volumes, titles, cmaps):
    remove_key_conflicts(['j', 'k'])
    figure, axes = plt.subplots(1, len(volumes), figsize=(10, 3.5), sharey=True)
    figure.suptitle('Multi-slice viewer')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.73, left=0.02, right=0.98)
    for j in range(len(axes)):
        axes[j].volume = volumes[j]
        axes[j].index = 0
        axes[j].axis('off')
        axes[j].imshow(volumes[j][axes[j].index], cmaps[j])
        axes[j].set_title(titles[j])
        # axes[j].set_axis_off()
    figure.text(0.05, 0.9, 'Press \'j\' for previous slice, \'k\' for next slice', ha='left', va='center', size=10)
    figure.text(0.95, 0.9, 'Slice 0', ha='right', va='center', size=12)
    figure.canvas.mpl_connect('key_press_event', on_key_press)


def remove_key_conflicts(new_keys):
    for p in plt.rcParams:
        if p.startswith('keymap.'):
            keys = plt.rcParams[p]
            remove_key_list = set(keys) & set(new_keys)
            for k in remove_key_list:
                keys.remove(k)


def on_key_press(e):
    """Bind events to key press."""
    figure = e.canvas.figure
    for ax in figure.axes:
        if e.key == 'j':
            prev_slice(figure, ax)
        elif e.key == 'k':
            next_slice(figure, ax)
        figure.canvas.draw()


def prev_slice(figure, ax):
    """Go to the previous slice."""
    volume = ax.volume
    if ax.index > 0:
        ax.index = (ax.index - 1)
        ax.images[0].set_array(volume[ax.index])
        for t in range(len(figure.texts)):
            if t > 1:
                figure.texts[t].set_visible(False)
        figure.text(0.95, 0.9, 'Slice ' + str(ax.index), ha='right', va='center', size=12)


def next_slice(figure, ax):
    """Go to the next slice."""
    volume = ax.volume
    if ax.index < volume.shape[0] - 1:
        ax.index = (ax.index + 1)
        ax.images[0].set_array(volume[ax.index])
        for t in range(len(figure.texts)):
            if t > 1:
                figure.texts[t].set_visible(False)
        figure.text(0.95, 0.9, 'Slice ' + str(ax.index), ha='right', va='center', size=12)


scrollable_slice_viewer([img_mat.T, prob_mat.T, pred_class.T, label_mat.T, error_map.T],
                   ['MRA img', 'Probability map', 'Predicted segmentation', 'Ground-truth', 'Error map'],
                   [cmap1, cmap2, cmap1, cmap1, cmap3])

plt.show()
print('DONE')
