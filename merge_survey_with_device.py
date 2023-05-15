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

def classify_multiple_sensor_data(
    exercises,
    sensor_data_paths,
    no_gait_in_between_exercises=False,
    masterfile='./Data/Back/JSON files/parkinsondb-f2b94-Patient-export.json',
    processed_path='./Data/Back/Processed/',
    create_images=False
    ):
    ''' 
        Classifies sensor data (sets) found at [sensor_data_paths] for all [exercises]. Expects this sensor data to be in the form (timestamp,x,y,z).

        [sensor_data_paths] can either be multiple paths to sensor data files or a folder. If it is a folder it classifies all the files from that folder.
        
        The classification is done through the corresponding survey data found in the [masterfile].

        Accepted exercises: walk, s2s

        [no_gait_in_between_exercise] accepts either a boolean, which defines whether all exercise have gait in between, or a array of booleans, which defines whether there is gait in between each individual exercise.
    '''
    if type(sensor_data_paths) != list:
        sensor_files = [ (sensor_data_paths + '\\' + f) for f in listdir(sensor_data_paths) if isfile(join(sensor_data_paths, f)) ]
        print('{} is seen as a folder. Number of files found: {}.'.format(sensor_data_paths,len(sensor_files)))
        sensor_data_paths = sensor_files
    
    number_of_paths = len(sensor_data_paths)
    number_of_errors = 0
    print('Starting classification of {} files.'.format(number_of_paths))

    max_iteration_time = 30 # seconds

    for idx,path in enumerate(sensor_data_paths):
        # if idx > 14:
            leon_utils.progressBar(idx,number_of_paths,40)
            try:
                with leon_utils.time_limit(max_iteration_time,('"Classification of file ' + path + '".')):
                    sensor_data = classify_sensor_data(exercises,path,masterfile,no_gait_in_between_exercises)
            except Exception as e:
                    number_of_errors += 1
                    print("Couldn't classify file ({}). Error message: {}.".format(path,e))
            try:
                sensor_data = aap.delete_unclassified_data(sensor_data)
                sensor_data['class'] = sensor_data['class'].apply(aap.classification_to_gait_map_integer)

                file_name = leon_utils.get_file_name(path)
                sensor_data.to_csv(processed_path + file_name,index = False)
                if create_images:
                    aap.visualize_gait_data(sensor_data,figure_name='images\\Classified data\\' + file_name)
            except Exception as e:
                    print("Data file seems strange. Classification gave no results ({}). Error message: {}.".format(path,e))


    print('Succesfully classified {} sensor data files. Not classified due to errors: {}.'.format(number_of_paths - number_of_errors,number_of_errors))
    

def readTimestampsForExercise( 
    exercise,
    filename,
    masterfile
    ):
    """
    Output: array of exercise specific data from the questionaire.

    For a [filename] corresponding to the data from an ax3 device, this function returns the annotation from the questionaire for the specified [exercise]. The timestamps are adjusted for summer time if necessary (through the summertime_correction function).



    Accepted exercises: walk, s2s
    """

    filename = leon_utils.get_basename(filename) + '.cwa'

    json_data = get_JsonData_from_masterfile(filename, masterfile)
    curr_android_key = get_AndroidKey(json_data, filename)
    
    # Get Exercise Information
    exercise_report = (
        json_data.get("AndroidReport").get(curr_android_key).get(exercise)
    )

    time_offset = json_data.get("AndroidReport").get(curr_android_key).get("time_offset")
    result = list()

    # S2S
    if exercise == "s2s":
        trial_details = [(exercise_report.get("aborted"),"npsts"),((exercise_report.get("aborted")),"fpsts")]
        for (aborted,details) in trial_details:
            info = list(range(3))
            info[0] = None
            if aborted:
                info[1] = None
                info[2] = None
            else:
                try:
                    corrected_timestamp = (exercise_report.get(details).get("startTimestamp") + time_offset) / 1000
                    # corrected_timestamp = summertime_correction(corrected_timestamp)
                    info[1] = corrected_timestamp 
                    info[2] = exercise_report.get(details).get("duration") / 1000
                except:
                    info[1] = None
                    info[2] = None

            result.append(info)
            
        result = pd.DataFrame(result,columns=['steps','start','duration'])

    # Walk
    if exercise == "walk":
        trial_details = [("first_steps","npw_first"),("second_steps","npw_second"),("third_steps","npw_third")]
        # TODO: print("What to do if this exercise is aborted? (walk)")

        for (steps,details) in trial_details:
            info = list(range(3))
            if exercise_report.get(steps) != None:
                info[0] = exercise_report.get(steps)
            else:
                info[0] = exercise_report.get(details).get(steps)
            
            corrected_timestamp = (exercise_report.get(details).get("startTimestamp") + time_offset) / 1000
            # corrected_timestamp = summertime_correction(corrected_timestamp)
            info[1] = corrected_timestamp 
            info[2] = exercise_report.get(details).get("duration") / 1000

            result.append(info)

        result = pd.DataFrame(result,columns=['steps','start','duration'])


    # Tug
    if exercise == "tug":
        print("Tug is not yet developed! Using it doesn't do anything but print this line.")  # TODO Some steps, but is it worth it?
        # if exercise_report.get("npt_first") != None:
        #     tsInit = exercise_report.get("npt_first").get("startTimestamp") + time_offset
        #     tsEnd = tsInit + exercise_report.get("npt_first").get("duration")
        #     result.append((tsInit,tsEnd))
        # if exercise_report.get("npt_second") != None:
        #     tsInit = exercise_report.get("npt_second").get("startTimestamp") + time_offset
        #     tsEnd = tsInit + exercise_report.get("npt_second").get("duration")
        #     result.append((tsInit,tsEnd))
        # if exercise_report.get("dttcog_first") != None:
        #     tsInit = exercise_report.get("dttcog_first").get("startTimestamp") + time_offset
        #     tsEnd = tsInit + exercise_report.get("dttcog_first").get("duration")
        #     result.append((tsInit,tsEnd))
        # if exercise_report.get("dttcog_second") != None:
        #     tsInit = exercise_report.get("dttcog_second").get("startTimestamp") + time_offset
        #     tsEnd = tsInit + exercise_report.get("dttcog_second").get("duration")
        #     result.append((tsInit,tsEnd))
        # if exercise_report.get("dttmotor_first") != None:
        #     tsInit = exercise_report.get("dttmotor_first").get("startTimestamp") + time_offset
        #     tsEnd = tsInit + exercise_report.get("dttmotor_first").get("duration")
        #     result.append((tsInit,tsEnd))
        # if exercise_report.get("dttmotor_second") != None:
        #     tsInit = exercise_report.get("dttmotor_second").get("startTimestamp") + time_offset
        #     tsEnd = tsInit + exercise_report.get("dttmotor_second").get("duration")
        #     result.append((tsInit,tsEnd))
        # return [0, 0, 0],result

    return result


def get_AndroidKey(
    json_data, 
    original_data_filename
    ):
    '''
    Obtains the correct android key for the requested file (specified by the [original_data_filename]) from the [json_data].
    '''
    
    lab_report_keys = json_data.get("LabReport")

    curr_android_key = ""

    # Get Android Key
    for key in list(lab_report_keys):
        curr_filename = leon_utils.replace_signs_to_allow_filename(lab_report_keys.get(key).get("filename"))
        if curr_filename == original_data_filename:
            curr_android_key = lab_report_keys.get(key).get("androidKey")
            break
    return curr_android_key


def get_JsonData_from_masterfile(
    filename,
    master_file
    ):
    '''
    Given the location of the [master_file], the survey data is returned for the file with [filename]
    '''

    json_fid = open(master_file, "r", encoding="utf-8") 
    json_data = json.load(json_fid)
    
    for key in json_data:
        lab_report = json_data[key].get('LabReport')
        lab_report_files = []
        if lab_report != None:
            for report in lab_report:
                r = lab_report[report]
                # print(leon_utils.replace_signs_to_allow_filename(r['filename']), '\t ---- \t', filename)
                if leon_utils.replace_signs_to_allow_filename(r['filename']) == filename:
                    survey = json_data[key]
                    break
    return survey
    

def get_JsonData(
    report_code, 
    location="./Data/Back/JSON files/"
    ):
    """
    Returns the JSON data found in the [report_code].json file in the [location] directory.
    """
    if type(report_code) == int:
        report_code = str(report_code)

    try:
        json_fid = open(
            location + report_code + ".json", "r", encoding="utf-8"
        ) 
        json_data = json.load(json_fid)
        return json_data
    except:
        print("No JSON file found in {} with name {}.json".format(location,report_code))


def readPatientCodes(path="Data\Back\\archive\Patient codes.csv"):
    '''
    Get all file details from the patient codes file
    '''
    with open(path, mode="r") as csvfile:
        arrayFiles = {}
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            newFilename = row["filename"].split(".cwa")[0]
            arrayFiles[newFilename] = row
        return arrayFiles

def get_FileCode_and_Name(
    filename, 
    FileInfo_dir=".\Data\Patient codes.csv"
    ):
    '''
    Get specific file details specified by [filename].
    '''
    # Get file with all the corresponding codes
    mapFileInfo = readPatientCodes(FileInfo_dir)

    fileInfo = mapFileInfo[filename]
    original_data_filename = fileInfo["newFilename"]
    report_code = fileInfo["code"]
    patient_name = fileInfo["name"]

    return report_code, original_data_filename, patient_name

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
                    if classifications[i] != "":
                        print("Classification {} was already classified as {}. This will now be overwritten with {}.".format(i,classifications[i],success))
                    classifications[i]=success
                if ((not to_classify) & no_gait_in_between):
                    if classifications[i] != "":
                        print("Classification {} was already classified as {}. This will now be overwritten with 'no gait'.".format(i,classifications[i]))
                    classifications[i]="no gait"

    
    return classifications


# ------ Main function -------
def classify_sensor_data(
    exercises,
    sensor_data_path,
    masterfile,
    no_gait_in_between_exercises=False
    ):
    '''
    Classifies sensor data found at [sensor_data_path] for all [exercises]. Expects this sensor data to be in the form (timestamp,x,y,z).

    The classification is done through the corresponding survey data found in the [masterfile].

    Accepted exercises: walk, s2s

    [no_gait_in_between_exercise] accepts either a boolean, which defines whether all exercise have gait in between, or a array of booleans, which defines whether there is gait in between each individual exercise.
    '''
    if (type(no_gait_in_between_exercises) != bool):
        if (len(exercises) != len(no_gait_in_between_exercises)):
            raise IndexError("Exercises and no_gait_in_between arrays should be of the same length. No_gait_in_between_exercises either accepts a boolean or an array of booleans representing an individual boolean for each exercise.") 
    
    try:
        sensor_data = pd.read_csv(sensor_data_path,header=None)
        sensor_data.columns = ['timestamp','x','y','z']

        sensor_data_size = len(sensor_data.index)
    except Exception as e:
        print(e)
        print("Data at {} not found or in incorrect format".format(sensor_data_path))
        return

    sensor_file_name = leon_utils.get_file_name(sensor_data_path)
    classifications  = pd.Series(np.zeros(sensor_data_size))
    for i in range(classifications.size):
        classifications[i] = ''

    for idx,exercise in enumerate(exercises):
        if type(no_gait_in_between_exercises) == list:
            no_gait_in_between_exercise = no_gait_in_between_exercises[idx]
        else: 
            no_gait_in_between_exercise = no_gait_in_between_exercises


        exercise_details = readTimestampsForExercise(exercise,sensor_file_name,masterfile)

        starts              = exercise_details['start'].values
        durations           = exercise_details['duration'].values
        values              = sensor_data['timestamp'].values
        success             = exercise

        classifications = classify_array(starts,durations,values,classifications,success,no_gait_in_between_exercise)

    sensor_data['class'] = classifications
    
    # print("\n Sensor data succesfully merged with the survey data for path: {}!".format(sensor_data_path))

    return sensor_data





def getPatientInfo(
    filename,
    JSON_folder="./reports/data/",
    patient_codes_path="./reports/file-map/inputFileInfo.csv",
    ): 
    """
    This function will get the corresponding steps for the exercise
    indetified at the end of the filename i.e. "_walk"

    Important for the data to be organized in the same way.
    """
    # Get exercise name
    aux = filename.split("_")
    exercise = aux[-1].split(".")[0]
    exercise = exercise.lower()

    # Get file name without the name of exercise at the end
    allFileSeg = aux[0] + "".join("_" + str(e) for e in aux[1:-1])

    # Get Report Code and Original Name
    report_code, original_data_filename, patient_name = get_FileCode_and_Name(
        allFileSeg, FileInfo_dir=patient_codes_path
    )

    # Get Json Data and AndroidKey
    json_data = get_JsonData(report_code, location=JSON_folder)
    curr_android_key = get_AndroidKey(json_data, original_data_filename)

    meta_data = json_data.get('AndroidReport').get(curr_android_key).get('metadata')
        
    patient_age = meta_data.get('age')
    if(patient_age == 0 or patient_age == None ):
        patient_age = json_data.get('age')
    
    patient_height = meta_data.get('height')
    if(patient_height == 0 or patient_height == None):
        patient_height = json_data.get('height')
    
    patient_weight = meta_data.get('weight')
    if(patient_weight == 0 or patient_weight == None):
        patient_weight = json_data.get('weight')
    
    patient_info = dict()
    patient_info['name'] = patient_name
    patient_info['age'] = patient_age
    patient_info['height'] = patient_height
    patient_info['weight'] = patient_weight
    patient_info['code'] = report_code
    
    return patient_info



# def summertime_correction_2019(timestamp):
#     SummerHourChange = 1553990401  # 31/03/2019 00:00:01 UNIX
#     WinterHourChange = 1572134401  # 27/10/2019 00:00:01 UNIX
#     if (timestamp >= SummerHourChange) & (timestamp <= WinterHourChange):
#         timestamp = timestamp + 3600
#     return timestamp

# def summertime_correction_2018(timestamp):
#     SummerHourChange = 1521939600
#     WinterHourChange = 1540692000
#     if (timestamp >= SummerHourChange) & (timestamp <= WinterHourChange):
#         timestamp = timestamp + 3600
#     return timestamp

# def summertime_correction(timestamp):
#     '''
#     This function changes the timestamps of a dataframe to incorporate inconsistencies due to summertime. Currently only takes in account the summertime of 2018 and 2019.
#     '''
#     correct_df = summertime_correction_2018(summertime_correction_2019(timestamp))
#     return correct_df
