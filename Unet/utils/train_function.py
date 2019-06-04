"""
File name: train_function.py
Author: Jana Rieger
Date created: 02/24/2018

This script contains a function that trains a model with given parameters and saves it.
"""

import time
from Unet import config
from Unet.utils.unet import get_unet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
from keras.models import load_model
from Unet.utils.metrics import avg_class_acc, dice_coef_loss, dice_coef
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import os
import pickle
import numpy as np

if config.TOP_LEVEL == './':
    os.chdir('../')


def train_and_save(train_X, train_y, val_X, val_y, patch_size, num_epochs, batch_size, lr, dropout, num_channels,
                   activation, final_activation, optimizer, loss, metrics, num_patches, factor_train_samples, mean, std,
                   threshold, rotation_range, horizontal_flip, vertical_flip, shear_range, width_shift_range,
                   height_shift_range):
    print('patch size', patch_size)
    print('number of epochs', num_epochs)
    print('batch size', batch_size)
    print('learning rate', lr)
    print('dropout', dropout)

    # create the name of current run
    run_name = config.get_run_name(patch_size, num_epochs, batch_size, lr, dropout,
                                   len(train_X) // num_patches, len(val_X) // num_patches)
    model_filepath = config.get_model_filepath(run_name)
    train_metadata_filepath = config.get_train_metadata_filepath(run_name)

    # if model does not exit, train it
    loading = False
    if not os.path.isfile(model_filepath):
        # -----------------------------------------------------------
        # CREATING MODEL
        # -----------------------------------------------------------
        model = get_unet(patch_size, num_channels, activation, final_activation, optimizer, lr, dropout, loss, metrics)

        # -----------------------------------------------------------
        # CREATING DATA GENERATOR
        # -----------------------------------------------------------
        # transforming images and masks together
        data_gen_args = dict(rotation_range=rotation_range,
                             horizontal_flip=horizontal_flip,
                             vertical_flip=vertical_flip,
                             shear_range=shear_range,
                             width_shift_range=width_shift_range,
                             height_shift_range=height_shift_range,
                             fill_mode='constant')
        X_datagen = ImageDataGenerator(**data_gen_args)
        y_datagen = ImageDataGenerator(**data_gen_args)

        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1
        X_datagen.fit(train_X, augment=True, seed=seed)
        y_datagen.fit(train_y, augment=True, seed=seed)

        X_generator = X_datagen.flow(train_X, batch_size=batch_size, seed=seed, shuffle=True)
        y_generator = y_datagen.flow(train_y, batch_size=batch_size, seed=seed, shuffle=True)

        # combine generators into one which yields image and label
        train_generator = zip(X_generator, y_generator)

        # sanity check, visualise augmented patches
        # pyplot.figure()
        # shift_plotting = 0
        # for i in range(0, 9):
        #     pyplot.suptitle('original patches')
        #     pyplot.subplot(330 + 1 + i)
        #     pyplot.imshow(train_X[i + shift_plotting].reshape(patch_size, patch_size))
        # pyplot.figure()
        # for i in range(0, 9):
        #     pyplot.subplot(330 + 1 + i)
        #     pyplot.imshow(train_y[i + shift_plotting].reshape(patch_size, patch_size))
        # pyplot.figure()
        # for X_batch in X_generator:
        #     pyplot.suptitle('flip, rotation 30°, shear 20°')
        #     for i in range(0, 9):
        #         pyplot.subplot(330 + 1 + i)
        #         pyplot.imshow(X_batch[i + shift_plotting].reshape(patch_size, patch_size))
        #     break
        # pyplot.figure()
        # for y_batch in y_generator:
        #     for i in range(0, 9):
        #         pyplot.subplot(330 + 1 + i)
        #         pyplot.imshow(y_batch[i + shift_plotting].reshape(patch_size, patch_size))
        #     break
        # pyplot.show()

        # -----------------------------------------------------------
        # TRAINING MODEL
        # -----------------------------------------------------------
        start_train = time.time()
        # keras callback for saving the training history to csv file
        csv_logger = CSVLogger(config.get_train_history_filepath(run_name))
        # training
        history = model.fit_generator(train_generator, validation_data=(val_X, val_y),
                                      steps_per_epoch=factor_train_samples * len(train_X) // batch_size,
                                      epochs=num_epochs,
                                      verbose=2, shuffle=True, callbacks=[csv_logger])

        duration_train = int(time.time() - start_train)
        print('training took:', (duration_train // 3600) % 60, 'hours', (duration_train // 60) % 60,
              'minutes', duration_train % 60,
              'seconds')

        # -----------------------------------------------------------
        # SAVING MODEL
        # -----------------------------------------------------------
        print('Saving model to ', model_filepath)
        model.save(model_filepath)

        print('Saving params to ', train_metadata_filepath)
        history.params['batchsize'] = batch_size
        history.params['dropout'] = dropout
        history.params['patch_size'] = patch_size
        history.params['learning_rate'] = lr
        history.params['loss'] = loss
        history.params['mean'] = mean  # mean used for training data centering
        history.params['std'] = std  # std used for training data normalization
        history.params['samples'] = factor_train_samples * len(train_X)
        history.params['val_samples'] = len(val_X)
        history.params['total_time'] = duration_train
        history.params['rotation range'] = rotation_range
        history.params['horizontal_flip'] = horizontal_flip
        history.params['vertical_flip'] = vertical_flip
        history.params['shear_range'] = shear_range
        history.params['width_shift_range'] = width_shift_range
        history.params['height_shift_range'] = height_shift_range
        results = {'params': history.params, 'history': history.history}
        with open(train_metadata_filepath, 'wb') as handle:
            pickle.dump(results, handle)

    else:  # model exists, load it
        # -----------------------------------------------------------
        # LOADING MODEL
        # -----------------------------------------------------------
        loading = True
        print('Loading  model from', model_filepath)
        model = load_model(model_filepath,
                           custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
        model.summary()
        with open(train_metadata_filepath, 'rb') as handle:
            history = pickle.load(handle)

    # -----------------------------------------------------------
    # PERFORMANCE MEASURES
    # -----------------------------------------------------------
    start_perf = time.time()
    train_y_pred_probs = model.predict(train_X, batch_size=batch_size, verbose=1)
    train_y_pred_class = (train_y_pred_probs > threshold).astype(
        np.uint8)  # convert from boolean to int, values 0 or 1
    val_y_pred_probs = model.predict(val_X, batch_size=batch_size, verbose=1)
    val_y_pred_class = (val_y_pred_probs > threshold).astype(
        np.uint8)  # convert from boolean to int, values 0 or 1

    print()
    print("Performance:")

    train_y_f = train_y.flatten()
    train_y_pred_probs_f = train_y_pred_probs.flatten()
    train_y_pred_class_f = train_y_pred_class.flatten()

    val_y_f = val_y.flatten()
    val_y_pred_probs_f = val_y_pred_probs.flatten()
    val_y_pred_class_f = val_y_pred_class.flatten()

    train_auc = roc_auc_score(train_y_f, train_y_pred_probs_f)
    train_acc = accuracy_score(train_y_f, train_y_pred_class_f)
    train_avg_acc, train_tn, train_fp, train_fn, train_tp = avg_class_acc(train_y_f, train_y_pred_class_f)
    train_dice = f1_score(train_y_f, train_y_pred_class_f)

    val_auc = roc_auc_score(val_y_f, val_y_pred_probs_f)
    val_acc = accuracy_score(val_y_f, val_y_pred_class_f)
    val_avg_acc, val_tn, val_fp, val_fn, val_tp = avg_class_acc(val_y_f, val_y_pred_class_f)
    val_dice = f1_score(val_y_f, val_y_pred_class_f)

    print('train auc:', train_auc)
    print('train acc:', train_acc)
    print('train avg acc:', train_avg_acc)
    print('train dice:', train_dice)

    print('val auc:', val_auc)
    print('val acc:', val_acc)
    print('val avg acc:', val_avg_acc)
    print('val dice:', val_dice)

    duration_perf = int(time.time() - start_perf)
    print('performance assessment took:', (duration_perf // 3600) % 60, 'hours', (duration_perf // 60) % 60,
          'minutes',
          duration_perf % 60,
          'seconds')
    if not loading:
        duration_total = history.params['total_time'] + duration_perf
    else:
        duration_total = history['params']['total_time'] + duration_perf
    print('total time:', (duration_total // 3600) % 60, 'hours', (duration_total // 60) % 60, 'minutes',
          duration_total % 60,
          'seconds')

    # -----------------------------------------------------------
    # SAVING RESULTS
    # -----------------------------------------------------------
    print('Saving training results to ', train_metadata_filepath)
    performance = {'train_true_positives': train_tp, 'train_true_negatives': train_tn,
                   'train_false_positives': train_fp, 'train_false_negatives': train_fn,
                   'train_auc': train_auc, 'train_acc': train_acc, 'train_avg_acc': train_avg_acc,
                   'train_dice': train_dice,
                   'val_true_positives': val_tp, 'val_true_negatives': val_tn,
                   'val_false_positives': val_fp, 'val_false_negatives': val_fn,
                   'val_auc': val_auc, 'val_acc': val_acc, 'val_avg_acc': val_avg_acc,
                   'val_dice': val_dice}
    if not loading:
        history.params['total_time'] = duration_total
        results = {'params': history.params, 'history': history.history, 'performance': performance}
    else:
        history['params']['total_time'] = duration_total
        results = {'params': history['params'], 'history': history['history'], 'performance': performance}
    with open(train_metadata_filepath, 'wb') as handle:
        pickle.dump(results, handle)
    print('________________________________________________________________________________')
