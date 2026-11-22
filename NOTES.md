Notes: will structure later

data from phisyonet 
using Sleep EDF database

 2 files 

 
** PSG Files**
 7 channels of time series data 
 - 2 are EEG (we focus here)
PSG = polysynography (PSG) sleep study, stores multi-channel physiological signals and clincal annotations collected during sleep.

PSG file is NOT a sinle file extention  (usually refers to raw signals like .edf/ EDF+ etc etc)
** Hypnograms **( the ground truth?) 
also has body temp ( we will ignore)

hypnogram: graph that plots sleep stages overtime across a period of sleep (ground truth, truth values for testing purposes)



packages that are in this project
 numpy - arrays efficients 
 matplot lib for visualisaiton
 scikit-learn for preprosses and train-test split and evaluation
 torch - for feep learning
mne - oos for eeg to load edf files 
- load anotations from hypnogram 
- timeseies dats from psg


TODO: you can bring back the eeg data file in the project folder
        just add it to gitgnore!!! :)