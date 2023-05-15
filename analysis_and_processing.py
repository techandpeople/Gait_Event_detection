import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import feature_extraction as fe


# Raw to input

def delete_unclassified_data(df,class_column='class'):
    '''
    For each row in the [df], checks whether the atribute [class_column] is empty or not. It expects [class_column] to have string format.
    '''
    unclassified_indices = df.index[df[class_column] == ''].tolist()
    trimmed_df = df.drop(index=unclassified_indices)
    return trimmed_df            

def add_information(df, visualize=False):
    '''
    Adds the vector magnitude (vm) to the sensor data frames. vm = x^2 + y^2 + z^2 (no square root is taken, should be but doesn't matter in this case)
    '''
    
    # df['x_times_y'] = df['x'] * df['y']
    # df['x_times_z'] = df['x'] * df['z']
    # df['y_times_z'] = df['y'] * df['z']
    df['vector_magnitude'] = df['x'] * df['x'] + df['y'] * df['y'] + df['z'] * df['z']

    if visualize:
        visualize_gait_data(df,vm_added=True)

def classification_to_gait_map_integer(classification):
    '''
    Classifies an [classification] to an integer defining it as gait if 1, and as non-gait if 0.
    '''
    if classification == 'walk':
        return 1
    elif classification == 's2s':
        return 0
    elif classification == 'no gait':
        return 0
    else:
        return 0

def classification_to_gait_map_boolean(classification):
    '''
    Classifies an [classification] to a boolean defining it as gait if True, and as non-gait if False.
    '''
    if classification == 'walk':
        return True
    elif classification == 's2s':
        return False
    elif classification == 'no gait':
        return False
    else:
        return False

def visualize_gait_data(df, figure_name='images/gait_data_figure',vm_added=False):
    '''
    Creates a figure (name = [figure_name].jpg) of the [df] data. Expects an x, a y, and a z column for the lines. Furthermore, looks for 'class' column. If this exists, the background of the plot is coloured for the gait areas.

    If [vm_added] is turned on, the vm is also drawn.
    '''

    line_x = plt.scatter(df['timestamp'],df['x'],c="blue")
    line_y = plt.scatter(df['timestamp'],df['y'],c="yellow")
    line_z = plt.scatter(df['timestamp'],df['z'],c="red")
    if vm_added:
        line_vector_magnitude = plt.scatter(df['timestamp'],df['vector_magnitude'],c="orange")
        plt.legend((line_x, line_y, line_z, line_vector_magnitude), ('label_x', 'label_y', 'label_z', 'label_vector_magnitude'))
    else: 
        plt.legend((line_x, line_y, line_z), ('label_x', 'label_y', 'label_z'))

    if 'class' in df.columns:
        gait = False
        begin = 0
        end = 0
        # ----------------
        if (type(df['class'].iloc[0])==bool):
            print(3)
            for index,row in df.iterrows():

                if (row['class'] & (not gait)):
                    begin = row['timestamp']
                    # print('Begin: ',begin)
                    gait = True
                elif ((not row['class']) & gait):
                    end = row['timestamp']
                    # print('End: ',end)
                    plt.axvspan(begin, end, alpha=0.3, color='green')
                    gait = False
        else:
            for index,row in df.iterrows():

                if ((row['class']==1) & (not gait)):
                    begin = row['timestamp']
                    # print('Begin: ',begin)
                    gait = True
                elif ((not (row['class']==1)) & gait):
                    end = row['timestamp']
                    # print('End: ',end)
                    plt.axvspan(begin, end, alpha=0.3, color='green')
                    gait = False
        # -----------

        if gait:
            end = max(df['timestamp'].values)
            plt.axvspan(begin, end, alpha=0.3, color='green')
            gait = False

    plt.savefig(figure_name + ".jpg")
    plt.clf()
    # print("Plot visualised and saved to {}.jpg.".format(figure_name))


# Windows

# Needs drastic improvement: For every feature all variances/means are calculated... First calculate all features, then extract the necessary variances/means
def depict_variances_with_means(windows,classes,image_addition=''):
    '''
    Very slow function that creates scatter plots for all [windows] with [classes]. The variances are put against the means for all the features (x,y,z,vm)
    '''
    
    features = ['x','y','z','vm']
    print('Depicting variances and means.')
    for i,feature in enumerate(features):
        print('Depicting {} ({} %).'.format(feature,((i/4) * 100)))
        variances = list()
        means = list()
        for idx,window in enumerate(windows):
            window = pd.DataFrame(window,columns=['timestamp','x','y','z'])
            add_information(window)
            var = fe.extract_variances(window)[i]
            mean = fe.extract_means(window)[i]
            variances.append(var)
            means.append(mean)

        cdict = {0: 'red', 1: 'blue'}
        plt.scatter(means,variances,c=classes)
        plt.xlabel('mean')
        plt.ylabel('variance')
        # plt.legend()
        plt.savefig("images/fe/{}{}.png".format(feature,image_addition))


# Results
def make_prob_from_percentage(perc):
    if perc < 10:
        prob = '0.0%d' % perc
    else:
        prob = '0.%d' % perc
    return prob
        

def autolabel(rects,ax,color='black'):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                make_prob_from_percentage(int(height * 100)),
                ha='center', va='bottom', rotation=45, color=color)

def autolabel2(rects1,rects2,ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect1,rect2 in zip(rects1,rects2):
        height1 = rect1.get_height()
        height2 = rect2.get_height()
        difference = height2 - height1
        improvement = height2 >= height1
        if improvement:
            text = make_prob_from_percentage(int(difference * 100))
            color = "green"
        else:
            text = make_prob_from_percentage(int(abs(difference) * 100))
            color = "red"
        ax.text(rect1.get_x() + rect1.get_width(), 1.05*max(height1,height2),
                text,
                ha='center', va='bottom', rotation=45, color=color)

def visualise_from_csv(general_evaluation_file,personalised_evaluation_file,image_addition):
    """
    Given a [general_evaluation_file] and a [personalised_evaluation_file], this function creates a bar chart similar to the one drawn by the testing.complete_test function.

    Use the [image addition] to give a name to the create barchart.
    """
    print("Starting visualisation of {} and {}.".format(general_evaluation_file,personalised_evaluation_file))

    general_evaluations     = pd.read_csv(general_evaluation_file)
    personalised_evaluations    = pd.read_csv(personalised_evaluation_file)
    patients = ["P0","P1","P3","P4","P5","P6","P7","P8","P9","P10","P11","P12","P13","P14","P16","P17","P18"]

    print("---")
    gen_acc_avg = round(sum(general_evaluations['average'].values)/len(general_evaluations['average'].values) * 100, 1)
    pers_acc_avg = round(sum(personalised_evaluations['average'].values)/len(personalised_evaluations['average'].values) * 100, 1)
    print("General evalutions accuracy ({} %) and personalised evalutions accuracy ({} %) difference (pers - gen): {} %.".format(gen_acc_avg,pers_acc_avg,round(pers_acc_avg - gen_acc_avg,1)))
    
    differences = (personalised_evaluations['average'].values - general_evaluations['average'].values) * 100
    print("Largest improvement: {} %. Largest deterioration: {} %.".format(round(max(differences),1),round(min(differences),1)))
    print("---")

    width = .4
    separation = width/2
    X_axis = np.arange(len(patients))
    X_axis1 = X_axis - separation
    X_axis2 = X_axis + separation

    plt.close()
    plt.rcParams["hatch.linewidth"] = 3.5
    # plt.ylim(0.6,1.0)
    bla1 = plt.bar(X_axis1,general_evaluations['average'].values,label="general",color="darkgrey",width=width, edgecolor='black', hatch=r"////")
    bla2 = plt.bar(X_axis2,personalised_evaluations['average'].values,label="personalised",color="lightgrey",width=width, edgecolor='black')
    # autolabel2(bla1,bla2,plt)
    plt.xticks(X_axis,patients,rotation=50)
    plt.legend(loc = "lower right",framealpha=.9)
    plt.ylabel("F-Score")
    plt.xlabel("Patients")
    plt.ylim(.6, 1.)
    plt.savefig('.\images\Results\{}.png'.format(image_addition), bbox_inches="tight")
    
    print("Visualisation of {} and {} complete.".format(general_evaluation_file,personalised_evaluation_file))

    return differences, gen_acc_avg, pers_acc_avg

# Results
def visualise_from_csv_overall(general_evaluation_file1,general_evaluation_file2,personalised_evaluation_file1,personalised_evaluation_file2,image_addition):
    """
    Given two [general_evaluation_file] and two [personalised_evaluation_file], this function creates a bar chart comparing the averages of all the results.

    Use the [image addition] to give a name to the create barchart.
    """
    print("Starting visualisation of ({},{}) and ({},{}).".format(general_evaluation_file1,personalised_evaluation_file1,general_evaluation_file2,personalised_evaluation_file2))

    general_evaluations1        = pd.read_csv(general_evaluation_file1)
    personalised_evaluations1   = pd.read_csv(personalised_evaluation_file1)
    general_evaluations2        = pd.read_csv(general_evaluation_file2)
    personalised_evaluations2   = pd.read_csv(personalised_evaluation_file2)

    print("---")
    gen_acc_avg1  = round(sum(general_evaluations1['average'].values)/len(general_evaluations1['average'].values) * 100, 1)
    pers_acc_avg1 = round(sum(personalised_evaluations1['average'].values)/len(personalised_evaluations1['average'].values) * 100, 1)
    gen_acc_avg2  = round(sum(general_evaluations2['average'].values)/len(general_evaluations2['average'].values) * 100, 1)
    pers_acc_avg2 = round(sum(personalised_evaluations2['average'].values)/len(personalised_evaluations2['average'].values) * 100, 1)
    print("1. General evalutions accuracy ({} %) and personalised evalutions accuracy ({} %) difference (pers - gen): {} %.".format(gen_acc_avg1,pers_acc_avg1,round(pers_acc_avg1 - gen_acc_avg1,1)))
    print("2. General evalutions accuracy ({} %) and personalised evalutions accuracy ({} %) difference (pers - gen): {} %.".format(gen_acc_avg2,pers_acc_avg2,round(pers_acc_avg2 - gen_acc_avg2,1)))
    gen_acc_std1  = np.std(general_evaluations1['average'].values)
    pers_acc_std1 = np.std(personalised_evaluations1['average'].values)
    gen_acc_std2  = np.std(general_evaluations2['average'].values)
    pers_acc_std2 = np.std(personalised_evaluations2['average'].values)
    print("1. General evalutions standard dev ({} %) and personalised evalutions standard dev ({} %) difference (pers - gen): {} %.".format(gen_acc_std1,pers_acc_std1,round(pers_acc_std1 - gen_acc_std1,1)))
    print("2. General evalutions standard dev ({} %) and personalised evalutions standard dev ({} %) difference (pers - gen): {} %.".format(gen_acc_std2,pers_acc_std2,round(pers_acc_std2 - gen_acc_std2,1)))
    
    differences1 = (personalised_evaluations1['average'].values - general_evaluations1['average'].values) * 100
    differences2 = (personalised_evaluations2['average'].values - general_evaluations2['average'].values) * 100
    print("Largest improvement: {} %. Largest deterioration: {} %.".format(round(max(differences1),1),round(min(differences1),1)))
    print("Largest improvement: {} %. Largest deterioration: {} %.".format(round(max(differences2),1),round(min(differences2),1)))
    print("---")

    width = .4
    separation = width/2
    labels = ["NN","CNN"]
    X_axis = np.arange(len(labels))
    X_axis1 = X_axis - separation
    X_axis2 = X_axis + separation

    plt.close()
    
    plt.rcParams["hatch.linewidth"] = 1
    # plt.ylim(0.6,1.0)
    bla1 = plt.bar(X_axis1,[gen_acc_avg1/100,gen_acc_avg2/100],yerr=[gen_acc_std1,gen_acc_std2],label="general",color="darkgrey",width=width, edgecolor='black', hatch=r"/")
    bla2 = plt.bar(X_axis2,[pers_acc_avg1/100,pers_acc_avg2/100],yerr=[pers_acc_std1,pers_acc_std2],label="personalised",color="lightgrey",width=width, edgecolor='black')
    autolabel(bla1,plt,color='black')
    autolabel(bla2,plt,color='black')
    plt.xticks(X_axis,labels,rotation=50)
    plt.ylabel("F-Score")
    plt.xlabel("Models")
    plt.legend(loc = "lower right",framealpha=.9)
    plt.savefig('.\images\Results\{}.png'.format(image_addition), bbox_inches="tight")
    
    print("Visualisation of {},{} and {},{} complete.".format(general_evaluation_file1,personalised_evaluation_file1,general_evaluation_file2,personalised_evaluation_file2))

    return differences1, gen_acc_avg1, pers_acc_avg1, differences2, gen_acc_avg2, pers_acc_avg2

    
def visualise_from_csv_single_file(file_name,image_addition='Barchart single file',masked=False):
    """
    Given a [file_name] , this function creates a bar chart similar to the one drawn by the testing.complete_test function but only for a single data set, so no comparison.

    Use the [image addition] to give a name to the create barchart.
    """
    print("Starting visualisation of {}.".format(file_name))

    data     = pd.read_csv(file_name)
    patients = ["P0","P1","P3","P4","P5","P6","P7","P8","P9","P10","P11","P12","P13","P14","P16","P17","P18"]

    print("---")
    data_acc_avg = round(sum(data['average'].values)/len(data['average'].values) * 100, 1)
    print("Evalutions accuracy ({} %)..".format(data_acc_avg))
    print("---")

    width = .4
    x = np.arange(len(patients))
    y = data['average'].values

    if masked:
        mask1 = y < 0.8
        mask2 = y >= 0.8

        plt.close()
        plt.bar(x[mask1], y[mask1], color = 'red')
        plt.bar(x[mask2], y[mask2], color = 'blue')
        # plt.ylim(0.6,1.0)
        plt.xticks(x,patients,rotation=50)
        plt.savefig('.\images\Results\{}.png'.format(image_addition))
        
        print("Visualisation of {} complete.".format(file_name))
    else:
        plt.close()
        bla = plt.bar(x,y,label="general",color="red",width=width)
        autolabel(bla,plt)
        plt.xticks(x,patients,rotation=50)
        plt.savefig('.\images\Results\{}.png'.format(image_addition))
        
        print("Visualisation of {} complete.".format(file_name))

    return data_acc_avg


def boxplot_multiple(general_evaluation_files,personalised_evaluation_files,labels,image_addition=''):

    if len(general_evaluation_files) != len(personalised_evaluation_files):
        raise ValueError("Length of [general_evaluation_files] ({}) should be same as length of [personalised_evaluation_files] ({}).".format(len(general_evaluation_files),len(personalised_evaluation_files)))

    differences = list()
    for idx,general_evaluation_file in enumerate(general_evaluation_files):
        general_evaluations     = pd.read_csv(general_evaluation_file)
        personalised_evaluations    = pd.read_csv(personalised_evaluation_files[idx])
        difference = (personalised_evaluations['average'].values - general_evaluations['average'].values) * 100
        differences.append(difference)

    plt.close()
    plt.boxplot(differences,labels=labels)
    plt.xticks(rotation=50)
    plt.axhline(0, c='black')
    plt.savefig('.\images\Results\Boxplot of differences ({}).png'.format(image_addition))
    
    
    





































































































# Attempt at visualizing multiple days separately
# -----------------------------------------------
# def visualize_gait_data(df, figure_name='images/gait_data_figure'):
#     '''
#     Creates a figure (name = [figure_name].jpg) of the [df] data. Expects an x, a y, and a z column for the lines. Furthermore, looks for 'class' column. If this exists, the background of the plot is coloured for the gait areas.
#     '''

#     start,end = df['timestamp'].values[0],df['timestamp'].values[-1]
#     days = list()
#     while start < end:
#         days.append(start)
#         start += 86400 # seconds in a day

#     for day in days:
#         day_df = df[df['timestamp'] >= day]
#         day_df = day_df[day_df['timestamp'] < (day + 86400)]
#         print(1)
#         if not day_df.empty:
#             line_x = plt.scatter(day_df['timestamp'],day_df['x'],c="blue")
#             line_y = plt.scatter(day_df['timestamp'],day_df['y'],c="yellow")
#             line_z = plt.scatter(day_df['timestamp'],day_df['z'],c="red")
#             plt.legend((line_x, line_y, line_z), ('label_x', 'label_y', 'label_z'))

#             if 'class' in day_df.columns:
#                 gait = False
#                 begin = 0
#                 end = 0
#                 # ----------------
#                 if (type(day_df['class'].iloc[0])==bool):
#                     for index,row in day_df.iterrows():

#                         if (row['class'] & (not gait)):
#                             begin = row['timestamp']
#                             # print('Begin: ',begin)
#                             gait = True
#                         elif ((not row['class']) & gait):
#                             end = row['timestamp']
#                             # print('End: ',end)
#                             plt.axvspan(begin, end, alpha=0.3, color='green')
#                             gait = False
#                 elif (type(day_df['class'].iloc[0])==np.int64):
#                     for index,row in day_df.iterrows():

#                         if ((row['class']==1) & (not gait)):
#                             begin = row['timestamp']
#                             # print('Begin: ',begin)
#                             gait = True
#                         elif ((not (row['class']==1)) & gait):
#                             end = row['timestamp']
#                             # print('End: ',end)
#                             plt.axvspan(begin, end, alpha=0.3, color='green')
#                             gait = False
#                 # -----------

#                 if gait:
#                     end = max(day_df['timestamp'].values)
#                     plt.axvspan(begin, end, alpha=0.3, color='green')
#                     gait = False

#             plt.savefig(figure_name + '_' + str(day) + ".jpg")
#     # print("Plot visualised and saved to {}.jpg.".format(figure_name))
