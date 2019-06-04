## How to use

#### Dependencies
- Tensorflow
- Keras
- Nibabel
- Scikit learn
- NumPy
- Python version 3.6

#### Prepare the data
1. For data augmentation use the script Unet/data_processing/data_augmentation.py. It automatically creates the folder Unet/model_data where it saves the masked images, masks and masked labels as nifti files under new working names and also their augmented versions (Flipping over vertical axis from previous version. If no working augmented patients files are set in config.py then no augmentation is done.). This script also splits the patients into train/val/test according to Unet/config.py file. 

2. For patch extraction use the script Unet/data_processing/patch_extraction.py. Running this script will automatically extracts random patches and save them to .npy files. Number of patches to extract and their sizes are defined in the Unet/config.py file.

Tree structure of Unet/model_data after data augmentation and patch extraction:
```
-Unet
 |
 ----model_data
     |
     ---- test 
     |    |
     |    ---- 0033_img.nii
     |    |
     |    ---- 0033_label.nii
     |    |
     |    ---- 0033_mask.nii
     |    |
     |    ---- ...
     |
     ---- train
     |    |
     |    ---- patch96
     |    |    |
     |    |    ---- 0005_96_img.npy
     |    |    |
     |    |    ---- 0005_96_label.npy
     |    |    |
     |    |    ---- ...    
     |    |
     |    ---- 0005_img.nii
     |    |
     |    ---- 0005_label.nii
     |    |
     |    ---- 0005_mask.nii
     |    |
     |    ---- ...
     |
     ---- val
     |    |
     |    ---- …
```

#### Train model
Use the script train_unet.py. Set the training parameters such as patch size, number of epochs, batch size ..., augmentation methods on top of the script. Also set whether you want to train on rough grid or fine grid. Rough grid means choosing randomly given number of learning rate and dropout combinations for each patch size and batch size set. Fine grid trains all possible combinations from the patch size, batch size, learning rate and dropout lists. The training parameter combinations patch size, number of epochs, batch size, learning rate and dropout rate are saved into the tuned_params.csv file in Unet/models.

The trained Keras models and metadata from training with the training parameters and results are stored in directory Unet/models. For each trained model there are two files model_... and train_metadata_... saved.

Example:  
```
- Unet
  |
  ---- models
       |
       ---- model_patchsize_96_epochs_10_batchsize_8_lr_0.0001_dropout_0.0_train_41_val_11.h5py
       |
       ---- train_metadata_patchsize_96_epochs_10_batchsize_8_lr_0.0001_dropout_0.0_train_41_val_11.pkl
       |
       ---- train_history_patchsize_96_epochs_10_batchsize_8_lr_0.0001_dropout_0.0_train_41_val_11.csv
       |
       ---- …
       |
       ---- tuned_params.csv
```

The metadata contains:
* params:
    * epochs
    * steps - number of batches per epoch
    * verbose
    * do_validation - boolean
    * metrics
    * batchsize
    * dropout
    * patch_size
    * learning_rate
    * loss
    * mean - mean for patch-wise normalization
    * std - standard deviation for patch-wise normalization
    * samples - number of training samples per epoch
    * val_samples - number of validation samples per epoch
    * totat_time - training time in seconds
    * rotation range
    * horizontal_flip
    * vertical_flip
    * shear_range
    * width_shift_range
    * height_shift_range
* history - history of losses and metrics from training per epoch
* performance 
    * train_true_positives
    * train_true_negatives
    * train_false_positives
    * train_false_negatives
    * train_auc
    * train_acc
    * train_avg_acc
    * train_dice
    * val_true_positives
    * val_true_negatives
    * val_false_positives
    * val_false_negatives
    * val_auc
    * val_acc
    * val_avg_acc
    * val_dice
    
#### Predict segmentation
Use the script predict_full_brain.py. Set the dataset you want the predictions to be generated for: train/val/test and set the model parameters such as patch size, number of epochs, batch size ... on top of the script. You can predict segmentations for multiple models from rough grid training (parameters read from the Unet/models/tuned_params.csv) or on fine grid.

The predicted probabilities are saved to Unet/results/dataset (dataset = train/val/test) as nifti file with the patient name and params of the trained model in the filename.

Example:  
```
- Unet
  |
  ---- results
       |
       ---- test
       |    |
       |    ---- test_probs_0029__patchsize_96_epochs_10_batchsize_8_lr_0.0001_dropout_0.0_train_41_val_11.nii
       |    |
       |    ---- ...
       |
       ---- val
            |
            ---- val_probs_0024__patchsize_96_epochs_10_batchsize_8_lr_0.0001_dropout_0.0_train_41_val_11.nii
            |
            ---- ...
       
```

#### Assess performance and save results
You can use the script Unet/save_results_to_csv.py which calculates 4 performance measures: AUC, ACC, AVG CLASS ACC and DICE. It saves the performance measures for training set and either validation or test set to the csv file according to what you set. The csv is saved to Unet/results/ folder. Example name: result_table_test_20180508-070125.csv 

For calculating the measures from the Taha's software EvaluateSegmentation.exe you can use the script Unet/evaluate_segmentation.py.
The results are saved to Unet/rasults/eval_segment/ folder. There are the xml files from the software saved, the csv table per patient with the summarized results from those xml files and also one csv file with averaged measures for all patient in a dataset (train/val/test).

#### Visualize predictions
Use Unet/visualize_full_brain_prediction.py for plotting predicted segmentation together with the input MRA image and ground truth label. Again you need to set the training parameters (patch size, number of epochs, batch size ...) on top of the file and working name of the patient for which you want to see the segmentation. There are two plots generated. 1. 3 different slices of the matrices. 2. a scrollable visualisation of the whole matrices.