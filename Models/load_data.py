import numpy as np
import os
import pickle
import re

from sklearn.preprocessing import StandardScaler
from src.constants import BASE_DIR, PROCESSED_DATA_DIR, TRAINED_MODEL_DIR, BASE_DATA_DIR, SPECIAL_DATA_DIR


def load_data(type='Base', scale=False):
    # Load the data files
    # Split the input by the capital letters
    inputs = re.findall('[A-Z][^A-Z]*', type)
    name = ''

    special = False
    if type == 'SynCoun' or type == 'Winrate' or type == 'TimePickWr' or type == 'PickWr':
        name += type
        special = True
    else:
        for prefix in ['Time', 'Pick', 'Perf', 'Aug', 'Base']:
            if prefix in inputs:
                name += prefix

    name += 'Matrices.npz'

    if 'Base' in type:
        dir = os.path.join(BASE_DIR, PROCESSED_DATA_DIR, BASE_DATA_DIR, name)
    elif special:
        dir = os.path.join(BASE_DIR, PROCESSED_DATA_DIR, SPECIAL_DATA_DIR, name)
    else:
        dir = os.path.join(BASE_DIR, PROCESSED_DATA_DIR, name)
    data = np.load(dir)

    # Set the matrices
    if type == 'SynCoun':
        syn_matrix, coun_matrix = data['syn_matrix'], data['coun_matrix']
        return syn_matrix, coun_matrix
    elif type == 'Winrate':
        wr_matrix = data['wr_matrix']
        return wr_matrix
    elif type == 'TimePickWr':
        pick_matrix = data['pick_matrix']
        time_matrix = data['time_matrix']
        return time_matrix, pick_matrix
    elif type == 'PickWr':
        pick_matrix = data['pick_matrix']
        return pick_matrix
    else:
        X_matrix, y_matrix, X_train, X_test, y_train, y_test = data['X_matrix'], data['y_matrix'], data['X_train'], \
                                                               data['X_test'], data['y_train'], data['y_test']
        # Scale features
        if scale == True:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        return X_matrix, y_matrix, X_train, X_test, y_train, y_test

def save_data(file_name, X_matrix, y_matrix, X_train, X_test, y_train, y_test):
    if 'Base' in file_name:
        dir = os.path.join(BASE_DIR, PROCESSED_DATA_DIR, BASE_DATA_DIR, file_name)
    else:
        dir = os.path.join(BASE_DIR, PROCESSED_DATA_DIR, file_name)
    np.savez(dir, X_matrix = X_matrix, y_matrix = y_matrix, X_train = X_train,
             X_test = X_test, y_train = y_train, y_test = y_test)

def load_model(file_name):
    with open(os.path.join(BASE_DIR, TRAINED_MODEL_DIR, file_name + 'Model.pkl'), "rb") as f:
        model = pickle.load(f)
    return model

def save_model(file_name, model):
    file_name += '.pkl'
    with open(os.path.join(BASE_DIR, TRAINED_MODEL_DIR, file_name), "wb") as f:
        pickle.dump(model, f)