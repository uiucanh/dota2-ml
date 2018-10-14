import json
import pandas as pd

from tqdm import tqdm
from src.constants import TOTAL_NUM_HEROES
from src.Models.load_data import *
from sklearn.model_selection import train_test_split

"""
This script is used to test whether the publicly provided roles by Valve
can affect the performances of models
"""


def get_roles():
    l = []
    for hero in json_file:
        for role in hero['roles']:
            l.append(role)
    return set(l)


def convert_df(matrix, roles):
    df = pd.DataFrame(matrix)
    column_names = ['Hero %s' % str(id) for id in range(0, 115)] +\
                   ['Rad_' + role for role in roles] + ['Dire_' + role for role in roles]

    df.columns = column_names
    return df

def add_roles_values(match):
    radiant_team = np.ndarray.tolist(np.where(X_matrix.iloc[match, 0:TOTAL_NUM_HEROES] == 1)[0])
    dire_team = np.ndarray.tolist(np.where(X_matrix.iloc[match, 0:TOTAL_NUM_HEROES] == -1)[0])

    for hero in radiant_team:
        hero_roles = json_file[hero]['roles']
        for role in hero_roles:
            X_matrix['Rad_' + role][match] += 1

    for hero in dire_team:
        hero_roles = json_file[hero]['roles']
        for role in hero_roles:
            X_matrix['Dire_' + role][match] += 1

if __name__ == '__main__':
    json_file = None
    try:
        with open(os.path.join(BASE_DIR, 'Models/heroes2.json')) as f:
            json_file = json.load(f)
    except:
        print("Could not open heroes.json file")
        exit()

    X_matrix, y_matrix, X_train, X_test, y_train, y_test = load_data('Base')
    # Get a set of all roles
    roles = get_roles()

    # Plus 19 for the roles
    extra_columns = np.zeros((len(X_matrix), len(roles) * 2))
    X_matrix = np.append(X_matrix, extra_columns, axis=1)

    # Convert to pandas DF
    X_matrix = convert_df(X_matrix, roles)

    for match in tqdm(range(len(X_matrix))):
        add_roles_values(match)
    X_train, X_test, y_train, y_test = train_test_split(X_matrix, y_matrix, test_size=0.2, random_state=1411)

    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    print(logreg.score(X_test, y_test))