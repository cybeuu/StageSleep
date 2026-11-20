Notes: will structure later

data from phisyonet 
using Sleep EDF database

 2 files 
 PSG Files
 7 channels of time series data 
 - 2 are EEG (we focus here)
 Hypnograms ( the ground truth?) 
also has body temp ( we will ignore)

packages that are in this project
 numpy - arrays efficients 
 matplot lib for visualisaiton
 scikit-learn for preprosses and train-test split and evaluation
 torch - for feep learning
mne - oos for eeg to load edf files 
- load anotations from hypnogram 
- timeseies dats from psg