# -*- coding: utf-8 -*-
"""
File name: threshold_analysis.py
Author: Michelle Livne
Date created: 06/12/2018

The goal of this script is to:
    1) Display calibration plots of the pulled Unet and half-Unet output
    2) Assuming the output is not calibrated, Find the optimal threshold on the classifier output to yield best F1-score (based on the validation-set)

"""

# Import the relevant modules:
import os, glob, sys
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve

# Define helper functions:
def calc_opt_f1(labels,probs):
    # The function gets the classifier's labels(=ground truth), probs (=scores, i.e. classifier output) and possible thresholds
    # And returns the best possible F1-score and the threshold that yielded it
    
    from sklearn.metrics import f1_score    
    thresholds = np.linspace(0.,1.,110) # downsample to avoid long calc. times
    opt_f1 = 0 # we assume the classifier can do better than this...
    for thresh in (thresholds):
        pred_class = probs>thresh
        F1 = f1_score(labels,pred_class)
        if F1>opt_f1:
            opt_f1 = F1
            opt_thresh = thresh    
    return opt_f1,opt_thresh


# Config the tested architecture: Unet of half-Unet
full_architecture = False # True = Unet / False = half-Unet

# set path to the data, based on the used architecture:
os.chdir('C:\\Users\\livnem\\OneDrive\\Charite\\Projects\\Jana_segmentation\\for_calibration')
if full_architecture:
    arctype = 'full'
    path = 'C:\\Users\\livnem\\OneDrive\\Charite\\Projects\\Jana_segmentation\\for_calibration\\full_architecture\\results\\val'
else:
    arctype = 'half'
    path = 'C:\\Users\\livnem\\OneDrive\\Charite\\Projects\\Jana_segmentation\\for_calibration\\half_architecture\\results\\val'
    
full_path = 'C:\\Users\\livnem\\OneDrive\\Charite\\Projects\\Jana_segmentation\\for_calibration\\full_architecture\\results\\val'

# set path to the data, extract the relevant data-filenames
val_labels_path = 'C:\\Users\\livnem\\OneDrive\\Charite\\Projects\\Jana_segmentation\\for_calibration\\ground_truth_for _calibration\\val'
test_labels_path = 'C:\\Users\\livnem\\OneDrive\\Charite\\Projects\\Jana_segmentation\\for_calibration\\ground_truth_for _calibration\\test'
val_label_files = sorted(os.listdir(val_labels_path))
test_label_files = sorted(os.listdir(test_labels_path))
val_all_files = sorted(os.listdir(path))
val_prob_files = os.listdir(path)

# Get the model filenames based on the extracted data-filenames. The models-filenames consist of the used hyperparameters for that calculated model
models = val_all_files[0:24] # Take only the first 24 files, because after 24, the models repeat for different validation sets
models[:] = [model[15:] for model in models] # extract the models names. Ignore the first 15 characters, that contain the validation-set pt. numbering

# For each model: concatenate flatten validation ground-truth labels to one vector
os.chdir(path)
model = models[0] # start at the first model
val_prob_files = glob.glob('*'+model) # take all the validation-set probability files calculated for this model

labels_full_vec = np.ndarray(0) # initiate the flatten vector
for [i,file] in enumerate(val_prob_files): # for each model, pull (cocatenate) all probability maps into one vector (=prob_full_vec) and pull (concatenate) all labels into one vector (=labels_full_vec)
    # concatenate flatten validation ground-truth labels to one vector for the specific model (classifier)
    os.chdir(val_labels_path)
    label_file = val_label_files[i]
    label_img = nib.load(label_file)
    label_mat = label_img.get_data()
    label_vec = label_mat.flatten()
    labels_full_vec = np.append(labels_full_vec, label_vec, axis=0) 
lables_full_vec = labels_full_vec.round() # Make sure the labels are binary


# Loop over the different models and make calibration plots for the pulled validation-sets per-model 
os.chdir(path)
best_f1=0
prob_full_vec = np.ndarray(0) # initiate the flatten vector
for [i,model] in enumerate(models): # For each model, calculate and plot the calibration plots and calculate best thresh based on precision-recall curves
    val_prob_files = glob.glob('*'+model)
    prob_full_vec = np.ndarray(0)
    for [i,file] in enumerate(val_prob_files):
        prob = nib.load(file)
        prob_mat = prob.get_data()
        prob_vec = prob_mat.flatten()
        prob_full_vec = np.append(prob_full_vec, prob_vec, axis=0) 
    
    # Sanity-check: make sure that the range of scores is [0,1]    
    if max(prob_full_vec)>1 or min(prob_full_vec)<0:
        sys.exit('The probability range of the validation set for the model was in the wrong range',min(prob_full_vec),max(prob_full_vec))
        
    # Make precision-recall plot for the model:
    average_precision = average_precision_score(labels_full_vec,prob_full_vec)
    precision,recall,_ = precision_recall_curve(labels_full_vec,prob_full_vec) 
    opt_f1,opt_thresh = calc_opt_f1(labels_full_vec,prob_full_vec)
    if opt_f1>best_f1:
        best_f1 = opt_f1
        best_thresh = opt_thresh
        best_model = model
    
    print('For the architecture:',arctype)
    print('The best F1 score on the validation set was:',best_f1, 'with threshold:',best_thresh)
    print('Model:',best_model)
    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
        
    # Make calibration plots:
    plt.figure(i)
    ax1 = plt.subplot2grid((3,1),(0,0),rowspan=2)
    ax2 = plt.subplot2grid((3,1),(2,0))
    ax1.plot([0,1],[0,1],"k:",label="Perfectly calibrated")
    
    fraction_of_positives, mean_predicted_value = \
            calibration_curve(labels_full_vec.round(), prob_full_vec, n_bins=10)
    
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-")
    
    ax2.hist(prob_full_vec, range=(0, 1), bins=10,
                 histtype="step", lw=2)
    
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')
    
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    
    plt.tight_layout()
    plt.show()    
    

## Since the classifier is clearly not calibrated, the threshold is found empirically on each validation set (pulled analysis)

# plot precision-recall curves for each model, and extract optimal threshold for maximal F1 score:
    
# extract best threshold for maximal Hausdorff distance*** ??? --> classify using best thresh --> calc Hausdorff dist. --> take best
    
# Take the model with best F1 score

