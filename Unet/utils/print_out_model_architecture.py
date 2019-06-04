"""
File name: print_out_model_architecture.py
Author: Jana Rieger
Date created: 02/28/2018

This script saves the model architecture scheme to Unet/utils/model.png file.
"""

from keras.utils import plot_model
from Unet.utils.unet import get_unet
from Unet import config

model = get_unet(64, config.NUM_CHANNELS, config.ACTIVATION,
                 config.FINAL_ACTIVATION, config.OPTIMIZER, 1e-4, 0.2, config.LOSS_FUNCTION,
                 config.METRICS)
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)
