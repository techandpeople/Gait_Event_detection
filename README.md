# Gait_event_detection
A programme to extract gait events from time to position data.

## Elements explanation

This repository contains 7 elements as of the 9th of April.

### Main.py

This programme is used to run the different functions defined in this repository. No special things should happen here.

### Data 

This folder contains all the data used or created. The [CSV files] folder contains the [CSV files] with each patient's movement sensor data. The [JSON files] folder contains each patient's survey data. The [Patient codes.csv] file contains all the information linking the files in the [JSON files] and [CSV files] folders. The [Processed] folder contains the created/processed files.

### images

Folder where images are saved.

### merge_survey_with_device.py

This file has one mayor function: classify_sensor_data(exercises,sensor_data_path,no_gait_in_between_exercises=False,JSON_folder=None,patient_codes_path=None)). For a CSV file containing movement sensor data (defined by the sensor_data_path), this function extracts all movement data. After that, the movement data that was measured during the specified exercises is classified as that exercise. 

The timestamps are given in epoch/system time. To convert: https://www.epochconverter.com/.

All the other functions in this file are aiding this main function.

### analysis_and_processing.py

This file contains 3 simple processing functions for the extracted classified data that resulted from the main function of the [merge_survey_with_device.py]. Furthermore it contains a visualisation function, that creates a graph of the classified data.

One function deletes all unclassified data. Two other functions translate the classifications to either gait or non-gait. This can either be done to boolean or integers (1/0), depending on the needs.

The visualisation functions plots the movement data together with a highlighted part for where the data is classified as gait.

### training.py

This file contains all functions related to creating a model for predicting the gait/non-gait class of movement sensor data. It contains 3 main functions. One cuts the movement data up into windows, one reformats the data into numpy arrays in order for it to be compatible with Tensorflow, and the last actually trains the model. 

The train_cnn_model(windows,classes,batch_size=5,detailed_printing=False) function works in a literal sense (it returns a trained model) but the model created is not yet tested nor optimized. 

### leon_utils.py

Contains some functions for overall use.

### Acknowledgements

This work was partially supported by the EU H2020 WideHealth project (grant agreement No 952279)
