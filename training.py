import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import leon_utils
import os
import time
import datetime
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from os import listdir
from os.path import isfile, join
from math import floor

import analysis_and_processing as aap
import feature_extraction as fe

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # To not print stupid GPU warning of Tensorflow
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras import Input, Model, optimizers, losses, utils as keras_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' # Put settings back to normal

# ----------- Personalisation ------------------

def retrain_nn_model(model,windows,classes,initial_learning_rate=0.02,detailed_printing=False,timed=False):
    
    time_start = datetime.datetime.now()
    
    features_from_windows = np.asarray(fe.extract_features(windows))
    classes = np.asarray(classes)
    
    print("Retraining NN model. Shape: ({},{}).".format(features_from_windows.shape,classes.shape))
    
    EPOCHS = 10
    STEPS_PER_EPOCH = floor(len(features_from_windows) / EPOCHS)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=EPOCHS,
        decay_rate=0.97,

        staircase=True)
    
    model.compile(
                    optimizer=optimizers.SGD(learning_rate=lr_schedule),
                    loss=losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    
    verbose_fit = 0
    if detailed_printing:
        # print('Model summary:')
        # model.summary()
        # keras_utils.plot_model(model, "images/histories/model-blocks.jpg", show_shapes=True)
        verbose_fit = 1
    
    history = model.fit(
        epochs=EPOCHS,
        x = features_from_windows,
        y = classes,
        validation_split = 0.2,
        verbose=verbose_fit
        )

    time_taken = datetime.datetime.now() - time_start
    if timed:
        print("Time taken for retraining: {}.".format(time_taken.seconds))

    return model
    
def retrain_cnn_model(model,windows,classes,initial_learning_rate=0.02,detailed_printing=False,timed=False):
    
    time_start = datetime.datetime.now()
    
    windows,classes = reformat_data_to_numpy_arrays(windows,classes,verbose=0)
    
    try:
        num_of_windows,num_of_timesteps_per_windows,num_of_dimensions = windows.shape
    except:
        print('Data is not well-shaped. Shape is {}. This should have three dimensions.'.format(windows.shape))

    input_shape = (num_of_timesteps_per_windows,num_of_dimensions)

    print("Retraining CNN model. Shape: ({},{}).".format(windows.shape,classes.shape))
    
    EPOCHS = 10
    STEPS_PER_EPOCH = floor(len(windows) / EPOCHS)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=EPOCHS,
        decay_rate=0.97,

        staircase=True)
    
    model.compile(
                    optimizer=optimizers.SGD(learning_rate=lr_schedule),
                    loss=losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    
    verbose_fit = 0
    if detailed_printing:
        # print('Model summary:')
        # model.summary()
        # keras_utils.plot_model(model, "images/histories/model-blocks.jpg", show_shapes=True)
        verbose_fit = 1
    
    history = model.fit(
        epochs=EPOCHS,
        x = windows,
        y = classes,
        validation_split = 0.2,
        verbose=verbose_fit
        )

    time_taken = datetime.datetime.now() - time_start
    if timed:
        print("Time taken for retraining: {}.".format(time_taken.seconds))

    return model

# ----------- Model evaluation ------------------

def evaluate_nn_model(model,windows,classes,all_metrics=False,add_confusion_matrix=False):
        '''
        Returns accuracy as default. If [all_metrics], returns accuracy, f_score, precision, recall (in that order).
        '''
        features_from_windows = np.asarray(fe.extract_features(windows))
        classes = np.asarray(classes)
    
        evaluations = model.evaluate(x=features_from_windows,y=classes)
        accuracy    = evaluations[1]

        if all_metrics:
            predictions = model.predict(features_from_windows)
            predictions[predictions <= 0.5] = 0
            predictions[predictions > 0.5] = 1

            f_score = f1_score(classes, predictions)
            precision = precision_score(classes, predictions)
            recall = recall_score(classes, predictions)

            if add_confusion_matrix:

                print('\n ----------\t Confusion matrix \t----------')
                print(confusion_matrix(classes,predictions,normalize='true'))

            return accuracy, f_score, precision, recall

        return accuracy

def evaluate_cnn_model(model,windows,classes,all_metrics=False,add_confusion_matrix=False):
        '''
        Returns accuracy as default. If [all_metrics], returns accuracy, f_score, precision, recall (in that order).
        '''
        windows,classes = reformat_data_to_numpy_arrays(windows,classes,verbose=0)
        
        evaluations = model.evaluate(x=windows,y=classes)
        accuracy    = evaluations[1]

        if all_metrics:
            predictions = model.predict(windows)
            predictions[predictions <= 0.5] = 0
            predictions[predictions > 0.5] = 1

            f_score = f1_score(classes, predictions)
            precision = precision_score(classes, predictions)
            recall = recall_score(classes, predictions)

            if add_confusion_matrix:

                print('\n ----------\t Confusion matrix \t----------')
                print(confusion_matrix(classes,predictions,normalize='true'))

            return accuracy, f_score, precision, recall

        return accuracy

# ----------- Models ------------------

def train_combination_model(windows,classes,detailed_printing=False):
    
    X_train, X_test, y_train, y_test = train_test_split(windows,classes,test_size=.3)
    
    train_features_from_windows     = np.asarray(fe.extract_features(X_train))
    test_features_from_windows      = np.asarray(fe.extract_features(X_test))
    
    train_windows,train_classes     = reformat_data_to_numpy_arrays(X_train,y_train)
    test_windows,test_classes       = reformat_data_to_numpy_arrays(X_test,y_test)
    
    try:
        num_of_windows,num_of_timesteps_per_windows,num_of_dimensions = train_windows.shape
    except:
        print('Data is not well-shaped. Shape is {}. This should have three dimensions.'.format(train_windows.shape))
    
    nn_input    = Input(shape=(len(train_features_from_windows[0])), name="Extracted features")
    features    = layers.Dense(3,activation='sigmoid', kernel_regularizer=regularizers.l2(0.003))(nn_input)
    features    = Model(inputs=nn_input,outputs=features)

    cnn_input   = Input(shape=(num_of_timesteps_per_windows,num_of_dimensions), name="Windows")
    windows     = layers.Conv1D(filters = 8, kernel_size = 64, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.003))(cnn_input)
    windows     = layers.MaxPooling1D(3)(windows)
    windows     = layers.Flatten()(windows)
    windows     = layers.Dense(3)(windows)
    windows     = Model(inputs=cnn_input,outputs=windows)

    combined    = layers.concatenate([features.output,windows.output])
    combined    = layers.Dense(1)(combined)

    model = Model(
        inputs=[features.input,windows.input],
        outputs=combined
        )
        
    verbose_fit = 0
    if detailed_printing:
        print('Model summary:')
        model.summary()
        keras_utils.plot_model(model, "images/histories/model-blocks.jpg", show_shapes=True)
        verbose_fit = 1

    EPOCHS = 100

    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=EPOCHS,
        decay_rate=0.97,

        staircase=True)
    
    model.compile(
                    optimizer=optimizers.SGD(learning_rate=lr_schedule),
                    loss=losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    history = model.fit(
        x = [train_features_from_windows,train_windows],
        y = train_classes,
        validation_split = 0.2,
        verbose=verbose_fit,
        epochs=EPOCHS
        )

    if detailed_printing:
        plt.close
        
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.legend(['accuracy','validation accuracy'])
        plt.savefig('images/histories/model_accuracy_evaluation.jpg')
        
        plt.close()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['loss','val_loss'])
        plt.savefig('images/histories/model_loss_evaluation.jpg')

        plt.close()

        print('\n ----------\t Evaluation \t----------')

        model.evaluate(x=[test_features_from_windows,test_windows],y=test_classes)

        predictions = model.predict([test_features_from_windows,test_windows])
        predictions[predictions <= 0.5] = 0
        predictions[predictions > 0.5] = 1
        
        f_score = f1_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)

        print("F-score: {}, precision: {}, recall: {}.".format(f_score,precision,recall))

        print('\n ----------\t Confusion matrix \t----------')
        print(confusion_matrix(y_test,predictions,normalize='true'))

    return model

def train_nn_model(windows,classes,detailed_printing=False,timed=False):
    
    time_start = datetime.datetime.now()
    
    features_from_windows = np.asarray(fe.extract_features(windows))
    classes = np.asarray(classes)

    X_train, X_test, y_train, y_test = train_test_split(features_from_windows,classes,test_size=.3)

    print("Training NN model. Number of features: {}, dataset size: {}, train size: {}.".format(len(X_train[0]),len(classes),len(X_train)))

    features_input = Input(
            shape=(len(features_from_windows[0])), name="Extracted features"
        )

    features = layers.Dense(4,activation='sigmoid', kernel_regularizer=regularizers.l2(0.003))(features_input)
    features = layers.Dense(1,activation='sigmoid',name='Output', kernel_regularizer=regularizers.l2(0.003))(features)

    model = Model(
        inputs=[features_input],
        outputs=[features]
        )
    
    verbose_fit = 0
    if detailed_printing:
        print('Model summary:')
        model.summary()
        keras_utils.plot_model(model, "images/histories/model-blocks.jpg", show_shapes=True)
        verbose_fit = 1
    
    EPOCHS = 100

    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=EPOCHS,
        decay_rate=0.97,

        staircase=True)

    model.compile(
                    optimizer=optimizers.SGD(learning_rate=lr_schedule),
                    loss=losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    history = model.fit(
        x = X_train,
        y = y_train,
        validation_split = 0.2,
        verbose=verbose_fit,
        epochs=EPOCHS
        )

    if detailed_printing:
        plt.close
        
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.legend(['accuracy','validation accuracy'])
        plt.savefig('images/histories/model_accuracy_evaluation.jpg')
        
        plt.close()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['loss','val_loss'])
        plt.savefig('images/histories/model_loss_evaluation.jpg')

        plt.close()

        model.evaluate(X_test,y_test)

        predictions = model.predict(X_test)
        predictions[predictions <= 0.5] = 0
        predictions[predictions > 0.5] = 1

        f_score = f1_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)

        print("F-score: {}, precision: {}, recall: {}.".format(f_score,precision,recall))

        print('\n ----------\t Confusion matrix \t----------')
        print(confusion_matrix(y_test,predictions,normalize='true'))

    time_taken = datetime.datetime.now() - time_start
    if timed:
        print("Time taken for training: {}.".format(time_taken.seconds))

    return model

def train_cnn_model(windows,classes,detailed_printing=False,timed=False):
    
    time_start = datetime.datetime.now()

    windows,classes = reformat_data_to_numpy_arrays(windows,classes,verbose=0)
    
    try:
        num_of_windows,num_of_timesteps_per_windows,num_of_dimensions = windows.shape
    except:
        print('Data is not well-shaped. Shape is {}. This should have three dimensions.'.format(windows.shape))

    input_shape = (num_of_timesteps_per_windows,num_of_dimensions)

    X_train, X_test, y_train, y_test = train_test_split(windows,classes,test_size=.3)

    print("Training CNN model. Shape: ({},{}).".format(X_train.shape,y_train.shape))
    
    model = models.Sequential([
        layers.Conv1D(filters = 8, kernel_size = 32, padding='same', activation='relu',input_shape=input_shape, kernel_regularizer=regularizers.l2(0.003)),
        layers.MaxPooling1D(3),
        layers.Dropout(rate = .2),
        layers.Conv1D(filters = 8, kernel_size = 16, padding='same', activation='relu',input_shape=input_shape, kernel_regularizer=regularizers.l2(0.003)),
        layers.MaxPooling1D(3),
        layers.Dropout(rate = .2),
        layers.Flatten(),
        layers.Dense(3, activation='relu'),
        layers.Dense(1, activation='relu')
        ])

    EPOCHS = 100

    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=EPOCHS,
        decay_rate=0.97,

        staircase=True)
    
    model.compile(
                    optimizer=optimizers.SGD(learning_rate=lr_schedule),
                    loss=losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    
    verbose_fit = 0
    if detailed_printing:
        print('Model summary:')
        model.summary()
        keras_utils.plot_model(model, "images/histories/model-blocks.jpg", show_shapes=True)
        verbose_fit = 1
    
    history = model.fit(
        epochs=EPOCHS,
        x = X_train,
        y = y_train,
        validation_split = 0.2,
        verbose=verbose_fit
        )
        
    if detailed_printing:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.legend(['accuracy','validation accuracy'])
        plt.savefig('images/histories/model_accuracy_evaluation.jpg')
        
        plt.close()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['loss','val_loss'])
        plt.savefig('images/histories/model_loss_evaluation.jpg')

        model.evaluate(X_test,y_test)

        predictions = model.predict(X_test)
        predictions[predictions <= 0.5] = 0
        predictions[predictions > 0.5] = 1
        
        f_score = f1_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)

        print("F-score: {}, precision: {}, recall: {}.".format(f_score,precision,recall))

        print('\n ----------\t Confusion matrix \t----------')
        print(confusion_matrix(y_test,predictions,normalize='true'))

    time_taken = datetime.datetime.now() - time_start
    if timed:
        print("Time taken for training: {}.".format(time_taken.seconds))

    return model

# ----------- Prepare data for training ------------------

def prepare_data_for_training_separated(classified_data_paths,stride=50,window_size=None,verbose=0,allow_ambiguous_windows=True):
    '''
        Prepares all the classified data for training. This includes reformatting data to windows (which includes adding padding), extracting the class for each window, and reformatting everything to numpy arrays. The results are an array with all windows and an array with all classes.
    
        [classified_data_paths] can either be multiple paths to sensor data files or a folder. If it is a folder it classifies all the files from that folder.
        
        Notice that the chosen [stride] and [window_size] should be correctly chosen. As we add padding to the data so that it is divisible by [stride], [window_size] should also be divisible by [stride], as explained here: https://cs231n.github.io/convolutional-networks/#conv.
    '''
    if (window_size % stride != 0):
        raise ValueError("The given window size ({}) is not divisible by the given stride ({}).".format(window_size,stride))
    
    if type(classified_data_paths) != list:
        classified_files = [ (classified_data_paths + '\\' + f) for f in listdir(classified_data_paths) if isfile(join(classified_data_paths, f)) and f != '.gitkeep' ]
        print('{} is seen as a folder. Number of files found: {}.'.format(classified_data_paths,len(classified_files)))
        classified_data_paths = classified_files
    
    number_of_paths = len(classified_data_paths)
    number_of_errors = 0
    print('Starting preparation of {} files.'.format(number_of_paths))
    all_windows = list()
    all_classes = list()

    for idx,path in enumerate(classified_data_paths):
        leon_utils.progressBar(idx,number_of_paths,40)
        try:
            data = pd.read_csv(path)
            aap.add_information(data,visualize=False)
            windows = data_to_windows(data,stride=stride,window_size=window_size,verbose=verbose)
            windows,classes = extract_class_per_window(windows,allow_ambiguous_windows)
            all_windows.append((path,windows))
            all_classes.append(classes)
        except Exception as e:
            number_of_errors += 1
            print("Couldn't prepare file ({}). Error message: {}.".format(path,e))

    print('Succesfully prepared {} classified data files. Not prepared due to errors: {}.'.format(number_of_paths - number_of_errors,number_of_errors))

    return all_windows,all_classes

def prepare_data_for_training(classified_data_paths,stride=50,window_size=None,verbose=0,allow_ambiguous_windows=True):
    '''
        Prepares all the classified data for training. This includes reformatting data to windows (which includes adding padding), extracting the class for each window, reformatting everything to numpy arrays, and undersampling the data set. The results are an array with all windows and an array with all classes.
    
        [classified_data_paths] can either be multiple paths to sensor data files or a folder. If it is a folder it classifies all the files from that folder.
        
        Notice that the chosen [stride] and [window_size] should be correctly chosen. As we add padding to the data so that it is divisible by [stride], [window_size] should also be divisible by [stride], as explained here: https://cs231n.github.io/convolutional-networks/#conv.
    '''
    if (window_size % stride != 0):
        raise ValueError("The given window size ({}) is not divisible by the given stride ({}).".format(window_size,stride))
    
    if type(classified_data_paths) != list:
        classified_files = [ (classified_data_paths + '\\' + f) for f in listdir(classified_data_paths) if isfile(join(classified_data_paths, f)) and f != '.gitkeep' ]
        print('{} is seen as a folder. Number of files found: {}.'.format(classified_data_paths,len(classified_files)))
        classified_data_paths = classified_files
    
    number_of_paths = len(classified_data_paths)
    number_of_errors = 0
    print('Starting preparation of {} files.'.format(number_of_paths))
    all_windows = list()
    all_classes = list()

    for idx,path in enumerate(classified_data_paths):
        leon_utils.progressBar(idx,number_of_paths,40)
        try:
            data = pd.read_csv(path)
            aap.add_information(data,visualize=False)
            windows = data_to_windows(data,stride=stride,window_size=window_size,verbose=verbose)
            windows,classes = extract_class_per_window(windows,allow_ambiguous_windows)
            all_windows += windows
            all_classes += classes
        except Exception as e:
            number_of_errors += 1
            print("Couldn't prepare file ({}). Error message: {}.".format(path,e))

    # Add resampling
    # all_windows,all_classes = oversample(all_windows,all_classes)
    all_windows,all_classes = undersample(all_windows,all_classes)

    print('Succesfully prepared {} classified data files. Not prepared due to errors: {}.'.format(number_of_paths - number_of_errors,number_of_errors))

    return all_windows,all_classes

# -------- Resampling -------------

def oversample(dfs,classes,verbose=1):
    '''
    Oversampling the minority class. Only works when there are 2 classes, 0 and 1.
    '''
    binary = [ x == 1 or x == 0 for x in classes ]
    if not (all(binary)):
        raise ValueError("Classes should only contain 0 and 1s.")
    
    proportion = np.mean(classes)
    if proportion >= 0.5:
        mayority_class = 1
    else:
        mayority_class = 0
    minority_class = 1 - mayority_class

    
    mayority_class_size = len([c for c in classes if c == mayority_class])
    minority_dfs = [df for i,df in enumerate(dfs) if classes[i] == minority_class]
    minority_class_size =len(minority_dfs)
    difference = mayority_class_size - minority_class_size
    if verbose == 1:
        print("Minority class: {} ({}). Mayority class: {} ({}). The minority class will be over sampled by randomly adding {} minority classes data points.".format(minority_class,minority_class_size, mayority_class, mayority_class_size,difference))

    for _ in range(difference):
        n = random.randint(0,minority_class_size - 1)
        classes.append(minority_class)
        dfs.append(minority_dfs[n])

    return dfs,classes

def undersample(dfs,classes,verbose=1):
    '''
    Undersampling the mayority class. Only works when there are 2 classes, 0 and 1.
    '''
    binary = [ x == 1 or x == 0 for x in classes ]
    if not (all(binary)):
        raise ValueError("Classes should only contain 0 and 1s.")
    
    proportion = np.mean(classes)
    if proportion >= 0.5:
        mayority_class = 1
    else:
        mayority_class = 0
    minority_class = 1 - mayority_class

    minority_dfs        = [df for i,df in enumerate(dfs) if classes[i] == minority_class]
    minority_classes    = [c for c in classes if c == minority_class]
    minority_class_size = len(minority_dfs)

    mayority_dfs        = [df for i,df in enumerate(dfs) if classes[i] == mayority_class]
    mayority_classes    = [c for c in classes if c == mayority_class]
    mayority_class_size = len(mayority_dfs)
    
    difference          = mayority_class_size - minority_class_size
    
    if verbose == 1:
        print("Minority class: {} ({}). Mayority class: {} ({}). The mayority class will be under sampled by randomly removing {} mayority class data points.".format(minority_class,minority_class_size, mayority_class, mayority_class_size,difference))

    random_integers     = random.sample(range(0,mayority_class_size - 1),minority_class_size)

    mayority_dfs        = [ mayority_dfs[idx]        for idx in random_integers ]
    mayority_classes    = [ mayority_classes[idx]    for idx in random_integers ]

    dfs         = mayority_dfs + minority_dfs
    classes     = mayority_classes + minority_classes

    return dfs,classes

# -------- Reformatting data to numpy array ----------------

def reformat_data_to_numpy_arrays(windows,classes,verbose=1):
    if (type(classes) == list):
        classes = np.asarray(classes)
    
    reformat_single_window = lambda df: df.drop(columns=['timestamp']).values
    new_windows = list(map(reformat_single_window,windows))

    new_windows = np.asarray(new_windows)

    if verbose > 0:
        print('Reformatted the windows to numpy arrays: {}.'.format(new_windows.shape))

    return new_windows,classes


# -------------- Data to windows --------------------

def extract_class_per_window(windows,allow_ambiguous_windows):
    classes = list()
    classless_windows = list()

    for window in windows:
        window_class = leon_utils.give_mode(window['class'].values)
        
        if (not (1 - window_class) in (window['class'].values)) or allow_ambiguous_windows: # If window_class is unique, accept it.
            classless_window = window.drop(columns=['class'])
            classless_windows.append(classless_window)

            classes.append(window_class)
    
    return classless_windows,classes

def data_to_windows(dataframe,stride=50,window_size=None,verbose=1):
    '''
    Split data up into different windows. These different windows can then be analysed. This method is explained will in (ING GAO, PEISHANG GU , QING REN , JINDE ZHANG, AND XIN SONG (2019)).

    Notice that the chosen [stride] and [window_size] should be correctly chosen. As we put padding to the [dataframe] so that it is divisible by [stride], [window_size] should also be divisible by [stride], as explained here: https://cs231n.github.io/convolutional-networks/#conv.
    '''
    
    if (type(dataframe) == str):
        dataframe = pd.read_csv(dataframe)

    df_length = dataframe['timestamp'].size

    if ((window_size == None) or (window_size == 0)):
        window_size = stride
    elif (window_size % stride != 0):
        raise ValueError("The given window size ({}) is not divisible by the given stride ({}).".format(window_size,stride))

    if verbose > 0:
        print("Dataframe (length: {}) has the following columns: {}.".format(df_length,dataframe.columns.values))

    # Add padding
    #--------------------
    excess_data = df_length % stride
    to_be_padded = stride - excess_data

    if (to_be_padded % 2 == 1):
        dataframe = add_single_padding_row(dataframe)  
        to_be_padded -= 1
        to_be_padded = to_be_padded / 2
    
    dataframe = add_padding(dataframe,to_be_padded)
    df_length = dataframe['timestamp'].size
    #--------------------

    if verbose > 0:
        print("Padding added to dataframe (new length: {}).".format(df_length))

    num_of_windows = int(((df_length - window_size) / stride) + 1) # As explained in https://cs231n.github.io/convolutional-networks/#conv

    if verbose > 0:
        print("Number of windows to be created: {}. Stride: {}, window size: {}.".format(num_of_windows,stride,window_size))

    windows = list()
    for i in range(num_of_windows):
        start = i * stride
        end = start + window_size
        windows.append(dataframe[start:end])

    return windows

def add_padding(df,padding_size,as_gait=False):
    '''
    Adds a padding of [padding_size] to both sides of the [df].

    If [as_gait] is True, the class will be gait.
    '''
    if type(padding_size) == float:
        padding_size = int(padding_size)
    
    for i in range(padding_size):
        df = add_single_padding_row(df,at_beginning=True,as_gait=as_gait)  
        df = add_single_padding_row(df,at_beginning=False,as_gait=as_gait)
    return df

def add_single_padding_row(df,at_beginning=True,as_gait=False):
    '''
    Adds a single padding row to [df]. First, looks to find the timestamp right before or after the first or last timestamp (depending on [at_beginning]). It will first calculate the distance between the bordering timestamps.

    If [as_gait] is True, the class will be gait.
    '''
    
    df_timestamps = df['timestamp'].values 
    if at_beginning:
        begin = df_timestamps[0]
        new_timestamp = begin - (df_timestamps[1] - begin) # Give first previous timestamp that is equally distant from next timestamp
    else:
        end = df_timestamps[-1]
        new_timestamp = end + (end - df_timestamps[-2]) # Give first next timestamp that is equally distant from previous timestamp

    data = []
    data.insert(0, {'timestamp': new_timestamp, 'x': 0, 'y': 0, 'z': 0, 'class': as_gait})

    if at_beginning:
        df = pd.concat([pd.DataFrame(data), df], ignore_index=True)
    else:
        df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
    # df = pd.concat([pd.DataFrame(data), df], ignore_index=True)
    aap.add_information(df)

    return df


# -------------- Windows to data --------------------

def combine_windows_to_complete_timeseries(windows,classes,window_size=None,stride=None,visualise=True):
    
    if (stride == None) or (stride == 0):
        raise Warning("You didn't specify the [stride]. A [stride] of 50 will be used. Notice that this is not necessarily the correct [stride].")
        stride = 50
    
    if (window_size == None) or (window_size == 0):
        raise Warning("You didn't specify the [window_size]. The [window_size] will match the stride. Notice that this is not necessarily the correct [window_size].")
        window_size = stride

    if (window_size % stride != 0):
        raise ValueError("The given window size ({}) is not divisible by the given stride ({}).".format(window_size,stride))

    windows_per_timestamp = window_size / stride

    merged_window = windows[0]
    merged_window.insert(5,'class',[classes[0]] * len(merged_window['timestamp']))
    merged_window.insert(6,'number_of_windows_passed',[1] * len(merged_window['timestamp']))

    for i in range(len(windows) - 1):
        leon_utils.progressBar(i,len(windows)-1,30)
        new_window = windows[i + 1]
        try: 
            new_window.insert(5,'class',[classes[i + 1]] * len(new_window['timestamp']))
            new_window.insert(6,'number_of_windows_passed',[1] * len(new_window['timestamp']))
        except Exception as error:
            print(new_window.columns)
            print(error)

        merged_window = add_two_windows(merged_window,new_window,window_size,stride)

    number_of_windows_per_timestamp = window_size / stride

    merged_window['class'] = merged_window['class'] / merged_window['number_of_windows_passed']
    merged_window = merged_window.drop(columns=['number_of_windows_passed'])
    merged_window.loc[merged_window['class'] <= 0.5,'class'] = 0
    merged_window.loc[merged_window['class'] > 0.5,'class'] = 1

    if visualise:
        aap.visualize_gait_data(merged_window,vm_added=True)

    return merged_window

def add_two_windows(window1,window2,window_size,stride):
    if window1['timestamp'].iloc[len(window1['timestamp']) - window_size + stride] != window2['timestamp'].iloc[0]:
        raise Exception("Windows don't follow up on each other. Begin timestamp of next window in window1: {}, begin timestamp window2: {}.".format(window1['timestamp'].iloc[window_size - stride],window2['timestamp'].iloc[0]))

    class1 = window1['class'].values
    class2 = window2['class'].values

    number_of_windows_passed1 = window1['number_of_windows_passed'].values
    number_of_windows_passed2 = window2['number_of_windows_passed'].values

    merged_window = pd.concat([window1,window2[(len(class2) - stride):len(class2)]])
    
    merged_class = np.concatenate((class1[0:-(window_size - stride)],[a + b for a,b in zip(class1[- (window_size - stride):len(class1)],class2[0:(len(class1)-stride)])],class2[-stride:len(class2)]), axis=None)
    merged_number_of_windows_passed = np.concatenate((number_of_windows_passed1[0:-(window_size - stride)],[a + b for a,b in zip(number_of_windows_passed1[- (window_size - stride):len(number_of_windows_passed1)],number_of_windows_passed2[0:(len(number_of_windows_passed1)-stride)])],number_of_windows_passed2[-stride:len(number_of_windows_passed2)]), axis=None)

    merged_window['class'] = pd.Series(merged_class)
    merged_window['number_of_windows_passed'] = pd.Series(merged_number_of_windows_passed)
    
    return merged_window

