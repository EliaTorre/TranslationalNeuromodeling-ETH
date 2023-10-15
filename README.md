# project_Group_H

You need the following Toolboxes and versions
MATLAB 2022b
Deep Learning Toolbox 14.5
Image Processing 11.6
Signal Processing 9.1
Statistics and Machine Learning 12.4
Bioinformatics 4.16.1
SPM12

## Classification without Embedding and with Laplacian Embedding
to run the two classification techniques above open the Project.mlx file and add SPM12 to the MATLAB Path

Important files:
* getTrialtype.m: script that return the indexes of std/dev trials according to the first approach. 
* accuracies_RAW.m: script that performs 10-fold stratified cross-validation accuracies evaluation for the dataset without embedding. 
* accuracies_RAW_perm: script that performs 10-fold stratified cross-validation accuracies evaluation for the dataset without embedding for 30 permuted labels models to assess chance-level. 
* plotting_RAW.m: script that plots the 30 permuted labels models as histograms, fits the gaussian curve, estimates p-values and returns plots.
* accuracies_LAPLACE.m, accuracies_LAPLACE_perm.m and plotting_LAPLACE.m perform the same function as the above but for laplacian embedded features. 


## DCM: most important files and instructions to run the code
to run the DCM code cd into the dcm subfolder and add SPM12 to the MATLAB path.

Important files:
* dcm/doDCMs.m: script that inverts all DCMs (this script takes many days to run) and saves the DCM params (one file per DCM model) into the folder output_dir. Set the data_dir in line 8 to the folder that contains the pre-processed EEG data.
* dcm/evalDCMs.m: script that trains the SVM and RF classifiers for all DCM models that it finds in output_dir. It performs the permutation test and saves results in a latex table.

## Convolutional Recurrent Attention Model (CRAM)
CRAM folder contains a jupyter notebook python script with the pipeline to run the model. 


## Neural HGF
neural_hgf folder contains matlab scripts for Neural HGF.
neural_hgf_training.m is the script to run to train a new neural net from scratch.
classifier_results.m is script to run to get classifier results on generative embedding from Neural HGF.
It assumes you have saved models of neural nets (neural_hgf_trained_foldnr_1_of_4 means saved model for fold 1 of 4fold CV). hgf_data_1_4 is processed data for this section (again 1st fold of 4 fold).
Other csv files are data and results. Other functions are helper functions for constructing the neural network.
