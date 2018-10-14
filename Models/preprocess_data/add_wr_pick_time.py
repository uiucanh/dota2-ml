from src.Models.load_data import *
from src.constants import TOTAL_NUM_HEROES
from sklearn.model_selection import train_test_split

import numpy as np


def fill_pick_matrix():
    print("Filling pick matrix: \n")

    # Iterating through the matrix
    for hero in range(TOTAL_NUM_HEROES):
        # Iterating through the possible pick orders for a hero
        for j in range(5):
            # Find matches where the hero is picked at the order j in both radiant and dire
            # and whether they won in those matches
            rad_pick_j = np.where(X_train[:, -11 + j] == hero)[0]
            for match in rad_pick_j:
                if y_train[match] == 1:
                    win_pick_matrix[hero, j] += 1

            dire_pick_j = np.where(X_train[:, -6 + j] == hero)[0]
            for match in dire_pick_j:
                if y_train[match] == -1:
                    win_pick_matrix[hero, j] += 1

            total_pick_matrix[hero, j] = len(rad_pick_j) + len(dire_pick_j)

    pick_matrix = win_pick_matrix / total_pick_matrix
    return pick_matrix

# Return the index given the match duration
def get_index(duration):
    if duration >= 15 and duration < 20:
        index = 0
    elif duration >= 20 and duration < 30:
        index = 1
    elif duration >= 30 and duration < 40:
        index = 2
    elif duration >= 40 and duration < 50:
        index = 3
    else:
        index = 4
    return index


def fill_time_matrix():
    print("Filling time matrix: \n")

    # Iterating through the matrix
    for hero_id in range(TOTAL_NUM_HEROES):
        # Find the matches where hero with hero_id is present
        matches = np.where(X_train[:, hero_id] != 0)[0]
        for match in matches:
            # Get the match duration in those matches and convert into matrix index
            duration = X_train[match][-1]
            index = get_index(duration)
            total_time_matrix[hero_id, index] += 1
            if y_train[match] == X_train[match][hero_id]:
                win_time_matrix[hero_id, index] += 1

    time_matrix = win_time_matrix / total_time_matrix
    return time_matrix

# Calculate the average time values for both teams
def process_match_duration(radiant_team, dire_team, time_m, index=None, duration=None):
    total_wr = []
    avg_team_wr = []

    if index is None:
        index = get_index(duration)

    for team in [radiant_team, dire_team]:
        for hero in team:
            total_wr.append(time_m[hero][index])
        avg_team_wr.append(np.mean(total_wr))
        total_wr = []

    return avg_team_wr[0], avg_team_wr[1]

# Calculate the average pick order values for both teams
def process_match_pick(match):
    rad_wr = []
    dire_wr = []
    for rad_pick_no in range(5):
        index = rad_pick_no
        hero_id = int(match[index])
        hero_wr = pick_matrix[hero_id][rad_pick_no]
        rad_wr.append(hero_wr)

    for dire_pick_no in range(5):
        index = 5 + dire_pick_no
        hero_id = int(match[index])
        hero_wr = pick_matrix[hero_id][dire_pick_no]
        dire_wr.append(hero_wr)

    return np.mean(rad_wr), np.mean(dire_wr)

# This function is used by the recommender system
def get_wr_for_team(team, pick_m):
    if len(team) <= 5:
        team_wr = []
        for hero in team:
            hero_pick_wr = pick_m[hero][team.index(hero)]
            team_wr.append(hero_pick_wr)
        return np.mean(team_wr)
    else:
        return 0


if __name__ == '__main__':
    # Create new matrices
    win_pick_matrix = np.zeros((TOTAL_NUM_HEROES, 5))
    total_pick_matrix = np.zeros((TOTAL_NUM_HEROES, 5))

    win_time_matrix = np.zeros((TOTAL_NUM_HEROES, 5))
    total_time_matrix = np.zeros((TOTAL_NUM_HEROES, 5))

    # Load the base data with match duration, pick order information
    X_matrix, y_matrix, X_train, X_test, y_train, y_test = load_data('TimePickBase')
    X_matrix2, y_matrix2, X_train2, X_test2, y_train2, y_test2 = load_data('TimePickPerfBase')

    # Filling the matrices
    pick_matrix = fill_pick_matrix()
    print("#" * 100)
    time_matrix = fill_time_matrix()
    print("#" * 100)

    # Add 4 new columns for the pick order and match duration feature,, 2 for each team
    extra_columns = np.zeros((len(X_matrix), 4))
    X_matrix = np.append(X_matrix, extra_columns, axis=1)
    X_matrix2 = np.append(X_matrix2, extra_columns, axis=1)

    for match in range(len(X_matrix)):
        # Get the heroes in both teams
        radiant_team = np.ndarray.tolist(np.where(X_matrix[match, 0:TOTAL_NUM_HEROES] == 1)[0])
        dire_team = np.ndarray.tolist(np.where(X_matrix[match, 0:TOTAL_NUM_HEROES] == -1)[0])
        duration = X_matrix[match][-5]

        # Do the match duration features
        rad_time_wr, dire_time_wr = process_match_duration(radiant_team, dire_team, time_matrix, duration=duration)
        X_matrix[match][-4] = X_matrix2[match][-4] = rad_time_wr
        X_matrix[match][-3] = X_matrix2[match][-3] = dire_time_wr

        # Do the pick order features
        rad_pick_wr, dire_pick_wr = process_match_pick(X_matrix[match][-15:-5])
        X_matrix[match][-2] = X_matrix2[match][-2] = rad_pick_wr
        X_matrix[match][-1] = X_matrix2[match][-1] = dire_pick_wr

    X_matrix = np.delete(X_matrix, [range(115, 126)], axis=1)
    X_matrix2 = np.delete(X_matrix2, [range(115, 126)], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_matrix, y_matrix, test_size=0.2, random_state=1411)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_matrix2, y_matrix2, test_size=0.2, random_state=1411)

    np.savez(os.path.join(BASE_DIR, PROCESSED_DATA_DIR, SPECIAL_DATA_DIR, 'TimePickWrMatrices'), time_matrix=time_matrix, pick_matrix=pick_matrix)
    save_data('TimePickMatrices', X_matrix, y_matrix, X_train, X_test, y_train, y_test)
    save_data('TimePickPerfMatrices', X_matrix2, y_matrix2, X_train2, X_test2, y_train2, y_test2)



