### Make data frame in Pandas

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_data(data_dir, label_file, verbose=False):
    if data_dir == 'object-detection-crowdai':
        df_files = pd.read_csv(os.path.join(data_dir, label_file), header=0)
        df_vehicles = df_files[(df_files['Label']=='Car') | (df_files['Label']=='Truck')].reset_index()
        df_vehicles = df_vehicles.drop('index', 1)
        df_vehicles['File_Path'] = data_dir+ '/' +df_vehicles['Frame']
        df_vehicles = df_vehicles.drop('Preview URL', 1)
    if data_dir == 'object-dataset':
        df_files = pd.read_csv(os.path.join(data_dir, label_file), sep=' ', header=None, usecols=[0,1,2,3,4,5,6])
        df_files.columns = ['Frame', 'xmin', 'xmax', 'ymin', 'ymax', 'ind', 'Label']
        df_vehicles = df_files[(df_files['Label'] == 'car') | (df_files['Label'] == 'truck')].reset_index()
        df_vehicles['File_Path'] = data_dir + '/' + df_vehicles['Frame']
        df_vehicles = df_vehicles.drop('index', 1)
        df_vehicles = df_vehicles.drop('ind', 1)

    if verbose == True:
        print(df_vehicles.head(5))
    return df_vehicles

def split_train_val(data, test_size=0.2):
    """
    Splits the csv containing driving data into training and validation

    :param dataframe:data_frame for split
    :return: train_split, validation_split
    """
    file_names = data['File_Path'].unique()
    driving_data = [row for row in file_names]

    train_names, val_names = train_test_split(driving_data, test_size=test_size, random_state=1)
    train_data = data.loc[data['File_Path'].isin(train_names)]
    val_data = data.loc[data['File_Path'].isin(val_names)]
    return train_data, val_data

if __name__ == '__main__':
    data_dir = 'object-dataset'
    data_frame = load_data(data_dir=data_dir, label_file='labels.csv',verbose=True)
    print(len(data_frame))