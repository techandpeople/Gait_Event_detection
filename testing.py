import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from gait_extractor import Extractor
import json
import random
from sklearn.model_selection import train_test_split

import merge_survey_with_device as mswd
import analysis_and_processing as aap
import training as tr
import feature_extraction as fe
import leon_utils


def testing_personalisation_with_visualisation(classified_data_folder,alignment_file,MODEL_TYPE,STRIDE,WINDOW_SIZE,DATA_TYPE,number_of_trials=5,timed=False,evaluation_types=["accuracy"],boxplot_difference=False):
    '''
    Function to test personalisation of models, and visualise it after. Uses the data from the folder specified by [classified_data_folder].

    Runs the model specified by [MODEL_TYPE] (either 'NN' or 'CNN') for windows of size [WINDOW_SIZE] and stride [STRIDE].

    Images are saved to the [.\images\Results\DATA_TYPE\] folder.

    The number of trials for which the models are run is specified by [number_of_trials] (default = 5).

    Use the settings [timed], [evaluation_types], and [boxplot_differences] to set whether you want to add the time, which evaluation types/metrics you want, and whether you want to visualise the boxplot of the differences of the general and personalised evaluations.
    '''
    print("Testing with visualistion of personalisation methods. Settings: \n -- Number of trials: {} \n -- Timed: {} \n -- Evaluation types: {} \n -- Boxplot of differences: {} \n".format(number_of_trials,timed,evaluation_types,boxplot_difference))

    general_evaluations, personalised_evaluations, patients = testing_personalisation(MODEL_TYPE,classified_data_folder,alignment_file,STRIDE,WINDOW_SIZE,DATA_TYPE,number_of_trials,timed)
    
    columns = [ 'trial {}'.format(i+1) for i in range(number_of_trials)]

    for i in range(len(evaluation_types)):
        evaluation_type = evaluation_types[i]
        
        generals = pd.DataFrame(general_evaluations[i],columns=columns)
        personaliseds = pd.DataFrame(personalised_evaluations[i],columns=columns)
        
        generals['average'] = generals['trial 1'] * 0
        personaliseds['average'] = personaliseds['trial 1'] * 0

        for k,col in enumerate(columns):
            generals['average'] = generals['average'] * (k/(k+1)) + generals[col] * (1/(k+1))
            personaliseds['average'] = personaliseds['average'] * (k/(k+1)) + personaliseds[col] * (1/(k+1))


        generals.to_csv(".\Data\{}\Results\{} generals ({}).csv".format(DATA_TYPE,MODEL_TYPE,evaluation_type))
        personaliseds.to_csv(".\Data\{}\Results\{} personaliseds ({}).csv".format(DATA_TYPE,MODEL_TYPE,evaluation_type))

        width = .4
        separation = width/2
        X_axis = np.arange(len(patients))
        X_axis1 = X_axis - separation
        X_axis2 = X_axis + separation

        plt.close()
        # plt.ylim(0.6,1.0)
        plt.bar(X_axis1,generals['average'].values,label="general",color="red",width=width)
        plt.bar(X_axis2,personaliseds['average'].values,label="personalised",color="green",width=width)
        plt.xticks(X_axis,patients,rotation=50)
        plt.legend(loc = "lower right",framealpha=.9)
        plt.savefig('.\images\Results\{}\Bar chart comparison {} ({}).png'.format(MODEL_TYPE,evaluation_type,DATA_TYPE))

    for i in range(len(evaluation_types)):
        if boxplot_difference:
            evaluation_type = evaluation_types[i]

            differences = personalised_evaluations[i] - general_evaluations[i]
            
            plt.close()
            plt.boxplot(differences,labels=patients)
            plt.xticks(rotation=50)
            plt.axhline(0, c='r')
            plt.savefig('.\images\Results\{}\Boxplot {} ({}).png'.format(MODEL_TYPE,evaluation_type,DATA_TYPE))
            differences = pd.DataFrame(differences,columns=columns)
            differences['average'] = differences['trial 1'] * 0

            for k,col in enumerate(columns):
                differences['average'] = differences['average'] * (k/(k+1)) + differences[col] * (1/(k+1))
            
            differences.to_csv(".\Data\{}\Results\Differences\{} ({}, {}).csv".format(DATA_TYPE,evaluation_type,MODEL_TYPE,DATA_TYPE))
        
def testing_personalisation(MODEL_TYPE,classified_data_folder,alignment_file,STRIDE,WINDOW_SIZE,DATA_TYPE,number_of_trials,timed):
    '''
    Function to test personalisation of models. Uses the data from the folder specified by [classified_data_folder].

    Runs the CNN model for windows of size [WINDOW_SIZE] and stride [STRIDE].

    The number of trials for which the models are run is specified by [number_of_trials] (default = 5)

    [alignment_file] is used to correctly order the patients.

    NOTE: No images are created!
    '''
    if MODEL_TYPE == 'CNN' or MODEL_TYPE == 'NN':
        print("Start testing of {} personalisation.".format(MODEL_TYPE))
        print("Configurations: Stride: {}, window size: {}, data type: {}, number of trials: {}.".format(STRIDE,WINDOW_SIZE,DATA_TYPE,number_of_trials))
    else:
        raise ValueError('Variable Model_type must either be NN or CNN. Current value: {}.'.format(MODEL_TYPE))
    
    windows,classes = tr.prepare_data_for_training_separated(classified_data_folder,stride=STRIDE,window_size=WINDOW_SIZE,verbose=0,allow_ambiguous_windows=False)
    windows,classes,patient_numbers = order_files(windows,classes,alignment_file,DATA_TYPE)

    number_of_files = len(windows)
   
    accuracy_generals           = list()
    f_score_generals            = list()
    precision_generals          = list()
    recall_generals             = list()
    generals                    = accuracy_generals, f_score_generals, precision_generals, recall_generals

    accuracy_personals          = list()
    f_score_personals           = list()
    precision_personals         = list()
    recall_personals            = list()
    personals                   = accuracy_personals, f_score_personals, precision_personals, recall_personals

    patients                    = list()

    for i in range(number_of_files):
        per_path,per_windows = windows[i]
        per_classes = classes[i]

        print('\n--------{}------------'.format(MODEL_TYPE))
        print("Personalisation effort ({}/{}) for patient {} with file path {}. ".format((i+1),number_of_files,patient_numbers[i],per_path))
        print('----------------------')
        patients.append(leon_utils.get_patient_name(per_path))

        gen_windows = list()
        gen_classes = list()
        for k in range(number_of_files):
            if k != i:
                k_path,k_windows = windows[k]
                gen_windows += k_windows
                gen_classes += classes[k]

        acc_gens        = list()
        f_score_gens    = list()
        precision_gens  = list()
        recall_gens     = list()
        gens            = acc_gens, f_score_gens, precision_gens, recall_gens

        acc_pers        = list()
        f_score_pers    = list()
        precision_pers  = list()
        recall_pers     = list()
        pers            = acc_pers, f_score_pers, precision_pers, recall_pers

        for idx in range(number_of_trials):
            print('--- {}: ({}/{})---'.format(i+1,idx+1,number_of_trials))
            successful = False
            while not successful:
                general_windows, general_classes    = tr.undersample(gen_windows, gen_classes,verbose=0)
                personal_windows, personal_classes  = tr.undersample(per_windows, per_classes,verbose=0)
                pers_win_train, pers_win_test, pers_class_train, pers_class_test = train_test_split(personal_windows,personal_classes,test_size=.3)

                if MODEL_TYPE == 'CNN':
                    model = tr.train_cnn_model(general_windows,general_classes,detailed_printing=False,timed=timed)
                    gen_evaluations = tr.evaluate_cnn_model(model,pers_win_test,pers_class_test,all_metrics=True)
                    if gen_evaluations[0] > .6:
                        successful = True
                        model = tr.retrain_cnn_model(model,pers_win_train,pers_class_train,initial_learning_rate=0.01,detailed_printing=False,timed=timed)
                        per_evaluations = tr.evaluate_cnn_model(model,pers_win_test,pers_class_test,all_metrics=True)

                        if per_evaluations[0] < .6:
                            print("{} retraining is stuck in local optimum, try again.".format(MODEL_TYPE))
                            successful = False
                    else: 
                        print("{} is stuck in local optimum, try again.".format(MODEL_TYPE))
                else:
                    model = tr.train_nn_model(general_windows,general_classes,detailed_printing=False,timed=timed)
                    gen_evaluations = tr.evaluate_nn_model(model,pers_win_test,pers_class_test,all_metrics=True)
                    if gen_evaluations[0] > .6:
                        successful = True
                        model = tr.retrain_nn_model(model,pers_win_train,pers_class_train,initial_learning_rate=0.01,detailed_printing=False,timed=timed)
                        per_evaluations = tr.evaluate_nn_model(model,pers_win_test,pers_class_test,all_metrics=True)

                        if per_evaluations[0] < .6:
                            print("{} retraining is stuck in local optimum, try again.".format(MODEL_TYPE))
                            successful = False
                    else: 
                        print("{} is stuck in local optimum, try again.".format(MODEL_TYPE))
            
            for k in range(4):
                gens[k].append(gen_evaluations[k])
                pers[k].append(per_evaluations[k])

        for k in range(4):
            generals[k].append(gens[k])
            personals[k].append(pers[k])

    print('\n\n general_evaluations accuracies:')
    print(generals[0])

    print('\n\n personalised_evaluations accuracies:')
    print(personals[0])

    return generals, personals, patient_numbers
        


def order_files(windows,classes,alignment_file,DATA_TYPE):
    '''
    Orders the [windows] and [classes] in accordance with the [aligntment_file] ordering of the patient numbers.
    '''
    
    alignments = pd.read_csv(alignment_file)

    paths = list(map(lambda x: leon_utils.get_file_name(x[0]),windows))

    ordered_windows = list()
    ordered_classes = list()
    patient_numbers = list()

    for idx,row in alignments.iterrows():
        if str(row['Time video']) == 'nan':
            continue
        path = row['File name ({})'.format(DATA_TYPE)]
        path_index = paths.index(path)
        
        ordered_windows.append(windows[path_index])
        ordered_classes.append(classes[path_index])
        patient_numbers.append('P{}'.format(row['Patient number']))

    ordered_paths = list(map(lambda x: x[0],ordered_windows))

    return ordered_windows,ordered_classes,patient_numbers
