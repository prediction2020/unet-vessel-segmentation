"""
File name: visualize_training.py
Author: Jana Rieger
Date created: 07/04/2018

This script plots the prediction of training patches.
"""

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from Unet import config
from Unet.utils.prepare_train_val_sets import create_training_datasets
from Unet.utils import helper
from Unet.utils.metrics import dice_coef_loss, dice_coef
import pickle
import os

################################################
# SET PARAMETERS
################################################
patch_size_list = [96]  # size of one patch n x n
num_epochs = 10  # number of epochs
batch_size_list = [8,16,32,64]  # list with batch sizes
learning_rate_list = [1e-4,1e-5]  # list with learning rates of the optimizer Adam
dropout_list = [0.0,0.1,0.2]  # percentage of weights to be dropped
num_patients_train = 41
num_patients_val = 11
save = False  # True for saving the plots into imgs folder as png files, False for not saving but showing the plots
################################################

if not os.path.exists(config.IMGS_DIR):
    os.makedirs(config.IMGS_DIR)

for patch_size in patch_size_list:
    for batch_size in batch_size_list:
        for lr in learning_rate_list:
            for dropout in dropout_list:
                print('________________________________________________________________________________')
                print('patch size', patch_size)
                print('batch size', batch_size)
                print('learning rate', lr)
                print('dropout', dropout)

                # create the name of current run
                run_name = config.get_run_name(patch_size, num_epochs, batch_size, lr, dropout, num_patients_train,
                                               num_patients_val)
                print(run_name)

                # -----------------------------------------------------------
                # LOADING MODEL DATA
                # -----------------------------------------------------------
                train_X, train_y, val_X, val_y, mean, std = create_training_datasets(patch_size, config.NUM_PATCHES,
                                                                                     config.PATIENTS)

                # -----------------------------------------------------------
                # LOADING MODEL
                # -----------------------------------------------------------
                model_filepath = config.get_model_filepath(run_name)
                model = load_model(model_filepath,
                                   custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
                # model.summary()

                train_metadata_filepath = config.get_train_metadata_filepath(run_name)
                with open(train_metadata_filepath, 'rb') as handle:
                    train_metadata = pickle.load(handle)

                print('Train params:')
                print(train_metadata['params'])
                print('Train performance:')
                print(train_metadata['performance'])

                # -----------------------------------------------------------
                # PREDICTION
                # -----------------------------------------------------------
                start = 495
                nr_predict = 10
                print('Prediction of training patches:')
                train_y_pred = model.predict(train_X[start:start + nr_predict], batch_size=1, verbose=1)
                print('Prediction of validation patches:')
                val_y_pred = model.predict(val_X[start:start + nr_predict], batch_size=1, verbose=1)

                # -----------------------------------------------------------
                # PLOTTING CURVES
                # -----------------------------------------------------------
                title = 'patchsize: ' + str(train_metadata['params']['patch_size']) + ', epochs: ' + str(
                    train_metadata['params']['epochs']) + ', batchsize: ' + str(
                    train_metadata['params']['batchsize']) + '\n' + ' lr: ' + str(
                    train_metadata['params']['learning_rate']) + ', dropout: ' + str(
                    train_metadata['params']['dropout']) + '\n' + ' number of training samples: ' + str(
                    train_metadata['params']['samples']) + ', number of validation samples: ' + str(
                    train_metadata['params']['val_samples'])

                save_name = config.IMGS_DIR + 'curves_' + run_name + '.png'
                helper.plot_loss_acc_history(train_metadata, save_name=save_name, suptitle=title, val=True, save=save)

                # -----------------------------------------------------------
                # PLOTTING PREDICTIONS
                # -----------------------------------------------------------

                # reshape for plotting
                train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2])
                train_y = train_y.reshape(train_y.shape[0], train_y.shape[1], train_y.shape[2])
                val_X = val_X.reshape(val_X.shape[0], val_X.shape[1], val_X.shape[2])
                val_y = val_y.reshape(val_y.shape[0], val_y.shape[1], val_y.shape[2])
                train_y_pred = train_y_pred.reshape(train_y_pred.shape[0], train_y_pred.shape[1], train_y_pred.shape[2])
                val_y_pred = val_y_pred.reshape(val_y_pred.shape[0], val_y_pred.shape[1], val_y_pred.shape[2])

                print('train_y_pred: max', np.max(train_y_pred), ' min', np.min(train_y_pred))
                print('val_y_pred: max', np.max(val_y_pred), ' min', np.min(val_y_pred))

                print('Plotting...')
                plt.figure(figsize=(12, 9))
                plt.suptitle(title)
                nr_patches_to_plot = nr_predict
                num_columns = 6

                for i in range(nr_patches_to_plot):
                    plt.subplot(nr_patches_to_plot, num_columns, (i * num_columns) + 1)
                    plt.title('train:' + '\n' + 'MRA img') if i == 0 else 0
                    plt.axis('off')
                    plt.imshow(train_X[start + i])

                    plt.subplot(nr_patches_to_plot, num_columns, (i * num_columns) + 2)
                    plt.title('train:' + '\n' + 'ground truth') if i == 0 else 0
                    plt.axis('off')
                    plt.imshow(train_y[start + i])

                    plt.subplot(nr_patches_to_plot, num_columns, (i * num_columns) + 3)
                    plt.title('train:' + '\n' + 'probability map') if i == 0 else 0
                    plt.axis('off')
                    plt.imshow(train_y_pred[i], cmap='hot')

                    cax = plt.axes([0.42, 0.05, 0.02, 0.75])
                    plt.colorbar(cax=cax)

                    plt.subplot(nr_patches_to_plot, num_columns, (i * num_columns) + 4)
                    plt.title('val:' + '\n' + 'MRA img') if i == 0 else 0
                    plt.axis('off')
                    plt.imshow(val_X[start + i])

                    plt.subplot(nr_patches_to_plot, num_columns, (i * num_columns) + 5)
                    plt.title('val:' + '\n' + 'ground truth') if i == 0 else 0
                    plt.axis('off')
                    plt.imshow(val_y[start + i])

                    plt.subplot(nr_patches_to_plot, num_columns, (i * num_columns) + 6)
                    plt.title('val:' + '\n' + 'probability map') if i == 0 else 0
                    plt.axis('off')
                    plt.imshow(val_y_pred[i], cmap='hot')

                    cax2 = plt.axes([0.9, 0.05, 0.02, 0.75])
                    plt.colorbar(cax=cax2)

                plt.subplots_adjust(bottom=0.05, right=0.9, top=0.8, left=0)
                if save:
                    plt.savefig(config.IMGS_DIR + 'preds_' + run_name + '.png')
                else:
                    plt.show()
print('DONE')
