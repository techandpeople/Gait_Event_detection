# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:07:11 2021

@author: leoni
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gait_extractor import Extractor
import json
import random

import merge_survey_with_device as mswd
import data_annotated_classification as dac
import analysis_and_processing as aap
import training as tr
import feature_extraction as fe
import testing


STRIDE              = 40
WINDOW_SIZE         = 200
DATA_TYPE           = 'Back'
MODEL_TYPE          = 'CNN'
number_of_trials    = 5
timed               = True
evaluation_types    = ['accuracy','f_score','precision','recall']
boxplot_difference  = False

# testing.testing_personalisation_with_visualisation(".\Data\{}\Processed".format(DATA_TYPE),".\Data\Annotation\Alignment.csv",MODEL_TYPE,STRIDE,WINDOW_SIZE,DATA_TYPE,number_of_trials,timed=timed,evaluation_types=evaluation_types,boxplot_difference=boxplot_difference)




# mswd.classify_multiple_sensor_data(exercises=['walk','s2s'],sensor_data_paths='Data\Back\CSV files',no_gait_in_between_exercises=[False,True],processed_path='./Data/Back/Processed/',create_images=True)


# windows,classes = tr.prepare_data_for_training(".\Data\{}\Processed".format(DATA_TYPE),stride=STRIDE,window_size=WINDOW_SIZE,verbose=0)

# aap.depict_variances_with_means(windows,classes,image_addition='-{}'.format(DATA_TYPE))

# print("Proportion gait data: {}. Data set size: {}.".format(np.mean(classes),len(classes)))

# model = tr.train_cnn_model(windows,classes,detailed_printing=True)




# dac.classify_through_video('Data\Annotation\Alignment.csv','File name ({})'.format(DATA_TYPE),'Data\Annotation\Patient files','Data\{}\CSV files'.format(DATA_TYPE),'Data\{}\Processed'.format(DATA_TYPE),'Back',preprocessing_file=".\Data\Preprocessing\Sensor faults.csv",create_images=True)
# aap.visualise_from_csv_single_file(".\images\Results\\NN\Test 5\\NN generals (f_score).csv")


# aap.visualise_from_csv(".\images\Results\\NN\Test 5\\NN generals (f_score).csv",".\images\Results\\NN\Test 5\\NN personaliseds (f_score).csv","NN Barchart (gebruikt)")
aap.visualise_from_csv(".\images\Results\\CNN\Test 3\\CNN generals (f_score).csv",".\images\Results\\CNN\Test 3\\CNN personaliseds (f_score).csv","CNN Barchart (gebruikt)")
aap.visualise_from_csv_overall(".\images\Results\\NN\Test 5\\NN generals (f_score).csv",".\images\Results\\CNN\Test 3\\CNN generals (f_score).csv",".\images\Results\\NN\Test 5\\NN personaliseds (f_score).csv",".\images\Results\\CNN\Test 3\\CNN personaliseds (f_score).csv","Overall comparison")





