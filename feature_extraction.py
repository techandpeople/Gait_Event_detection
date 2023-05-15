import pandas as pd
import numpy as np



def extract_means(df):
    '''
    For a sensor data frame [df] this function returns the means of the features x,y,z, and vm (in that order).
    '''
    
    xs = df['x'].values
    ys = df['y'].values
    zs = df['z'].values
    vms = df['vector_magnitude'].values / max(df['vector_magnitude'].values)

    x_mean = np.mean(xs)
    y_mean = np.mean(ys)
    z_mean = np.mean(zs)
    vm_mean = np.mean(vms)

    return x_mean,y_mean,z_mean,vm_mean

def extract_variances(df):
    '''
    For a sensor data frame [df] this function returns the variances of the features x,y,z, and vm (in that order).
    '''

    xs = df['x'].values
    ys = df['y'].values
    zs = df['z'].values
    if max(abs(df['vector_magnitude'].values)) == 0:
        vms = df['vector_magnitude'].values
    else:
        vms = df['vector_magnitude'].values / max(abs(df['vector_magnitude'].values))

    x_variance = np.var(xs)
    y_variance = np.var(ys)
    z_variance = np.var(zs)
    vm_variance = np.var(vms)

    return x_variance,y_variance,z_variance,vm_variance


def extract_features_single_df(df):
    means = extract_means(df)
    variances = extract_variances(df)

    return np.asarray([means[0],means[1],means[2],means[3],variances[0],variances[1],variances[2],variances[3]])

def extract_features(dfs):
    feature_list = list(map(extract_features_single_df,dfs))

    return feature_list
