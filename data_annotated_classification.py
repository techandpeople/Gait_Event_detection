import re
import csv
import json
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import datetime, time

import analysis_and_processing as aap
import leon_utils

def get_duration(start,end,allow_negative_duration=False):
    '''
    Get time difference (duration) based on minutes and seconds (min:sec). Minutes and seconds should be in strings

    When [allow_negative_duration] is set to True, the time difference is given.
    '''
    s_min,s_sec = int(start.split(':')[0]),int(start.split(':')[1])
    e_min,e_sec = int(end.split(':')[0]),int(end.split(':')[1])
    duration = (e_min - s_min) * 60 + (e_sec - s_sec)

    if duration <= 0 and not allow_negative_duration:
        raise ValueError("Start ({}) is equal or larger than end ({}). Duration ({}) must be larger than 0.".format(start,end,duration))

    return duration

def classify(
    starts,
    durations,
    value
    ):
    '''
    Classifies the given [value]. If it is within the ([start],[start] + [duration]) intervals, it returns True, else False.
    '''

    if len(starts) != len(durations):
        raise IndexError("Starts and durations must be same length")
    
    c = False
    for i in range(len(starts)):
        start = starts[i]
        if ((value >= start) & (value <= start + durations[i])):
            c = True
    
    return c

def get_exercise_area(
    starts,
    durations,
    margin=2
    ):
    '''
    Given the [starts] and [durations] of an exercise, returns the exercise area, i.e. the beginning of the exercise - the [margin], and the duration of the exercise + the [margin]. The margin is in seconds (default 2).
    '''
    first_exercise_start = min(starts)
    begin_exercise_area = first_exercise_start - margin

    last_exercise_start = max(starts)
    last_exercise_start_arg = np.argmax(starts)
    end_last_exercise = last_exercise_start + durations[last_exercise_start_arg]
    end_exercise_area = end_last_exercise + margin
    duration_exercise = end_exercise_area - begin_exercise_area

    return begin_exercise_area,duration_exercise

def classify_array(
    starts,
    durations,
    values,
    classifications,
    success,
    no_gait_in_between=False
    ):
    '''
    Classifies the given [values]. For each value in [values] checks whether classified by classify function. If classified, changes the corresponding position in the [classifications] array to [success].

    If [no_gait_in_between], the time around the actual active exercise is classified as 'no gait'.
    '''
    
    if len(values) != len(classifications):
        raise IndexError("Values and classifications must be same length")

    try:
        exercise_start,exercise_duration = get_exercise_area(starts,durations)
    except:
        # print('Exercise not found: {}.'.format(success))
        exercise_start,exercise_duration = None,None


    if exercise_start != None:
        for i in range(len(values)):
            within_exercise_area = classify([exercise_start],[exercise_duration],values[i])
            if within_exercise_area:
                to_classify = classify(starts,durations,values[i])
                if (to_classify):
                    if classifications[i] != 0:
                        print("Classification {} was already classified as {}. This will now be overwritten with {}.".format(i,classifications[i],success))
                    classifications[i]=success
                if ((not to_classify) & no_gait_in_between):
                    if classifications[i] != 0:
                        print("Classification {} was already classified as {}. This will now be overwritten with 'no gait'.".format(i,classifications[i]))
                    classifications[i]=0

    
    return classifications

def trim_data(begin,end,data):
    data = data[data['timestamp'] < end]
    data = data[data['timestamp'] > begin]
    
    return data
    
def convert_to_timestamp(time_video,aligned_time_sensor,time):
    alignment_difference = get_duration(time_video,time,allow_negative_duration=True)
    converted_time =alignment_difference + aligned_time_sensor + .5 # Adding .5 as the manual alignment is done on whole seconds.
    return converted_time

def classify_through_video(
    alignment_file,
    alignment_file_column,
    annotation_files,
    sensor_files,
    processed_files,
    DATA_TYPE,
    preprocessing_file=None,
    create_images=False
    ):

    alignments = pd.read_csv(alignment_file,header=0)

    if preprocessing_file != None:
        preprocessing = pd.read_csv(preprocessing_file,header=0)

    errors = 0

    for idx,row in alignments.iterrows():
        # if idx < 3:
            leon_utils.progressBar(idx + 1,len(alignments.index),40)
            
            if str(row['Time video']) == 'nan':
                continue
                
            try:
                sensor_data_path = sensor_files + '\\' + row[alignment_file_column]
                sensor_data = pd.read_csv(sensor_data_path,header=None)
                sensor_data.columns = ['timestamp','x','y','z']
            except Exception as e:
                print(e)
                print("Data at {} not found or in incorrect format".format(sensor_data_path))
                errors += 1
                continue

            annotations = pd.read_csv('{}\\Annotation P{}.csv'.format(annotation_files,row['Patient number']))

            time_video, timestamp_sensor = row['Time video'],row['Timestamp sensor (without summertime)']
            
            gait_annotations = annotations[annotations['Gait'] == 'Yes']

            for idx_2,row_2 in gait_annotations.iterrows():
                gait_annotations.at[idx_2,'Until'] = get_duration(row_2['From'],row_2['Until'])
                gait_annotations.at[idx_2,'From'] = convert_to_timestamp(time_video,timestamp_sensor,row_2['From'])
            gait_annotations = gait_annotations.rename(columns = {'Until':'durations', 'From' : 'starts'})
            
            # ------------------------------------------------------
            sensor_data_size    = len(sensor_data.index)
            classifications     = pd.Series(np.zeros(sensor_data_size))
            
            starts              = gait_annotations['starts'].values
            durations           = gait_annotations['durations'].values
            values              = sensor_data['timestamp'].values

            classifications     = classify_array(starts,durations,values,classifications,success=1,no_gait_in_between=True)

            sensor_data['class'] = classifications

            # ------------------------------------------------------
            num_annotations     = annotations['From'].size - 1
            begin               = convert_to_timestamp(time_video,timestamp_sensor,annotations.at[0,'From']) 
            end                 = convert_to_timestamp(time_video,timestamp_sensor,annotations.at[num_annotations,'Until']) 
            
            sensor_data         = trim_data(begin,end,sensor_data)
            
            # ------------------------------------------------------
            if preprocessing_file != None and DATA_TYPE == 'Back':
                preprocessing_row = preprocessing.loc[preprocessing['Patient number'] == 'P{}'.format(row['Patient number'])].iloc[0]
                if preprocessing_row['Switch y and x'] == 'x':
                    # print('Switching y and x for patient {}.'.format(row['Patient number']))
                    sensor_data[['y','x']] = sensor_data[['x','y']]
                if preprocessing_row['flip'] == 'x':
                    # print('Flipping x feature of patient {}.'.format(row['Patient number']))
                    sensor_data['x'] = sensor_data['x'] * -1

            # ------------------------------------------------------
            sensor_data.to_csv(processed_files + '//' + row[alignment_file_column],index = False)
            
            if create_images:
                try:
                    aap.visualize_gait_data(sensor_data,figure_name='images\\Classified data\\' + DATA_TYPE + '\\' + row[alignment_file_column])
                except Exception as e:
                    print('\n\n ------------------ \n')
                    print(e)
                    print(row)
                    errors += 1
        
    print('Succesfully classified {} sensor data files. Not classified due to errors: {}.'.format(len(alignments.index) - errors,errors))
    


