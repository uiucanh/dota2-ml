import itertools
from tqdm import tqdm
from src.Models.load_data import *
from src.constants import BASE_DIR, TOTAL_NUM_HEROES, PROCESSED_DATA_DIR
from sklearn.model_selection import train_test_split

"""
This script is a modified version of synergy and counter where the synergy
matrix is 3D
"""

# This function gets the synergy and counter percentage of hero i and hero j
# Iterate through all the recorded matches
# Find matches where hero i and hero j are present
# Check if they are on the same team and if they won
# Return percentage of win / total
def process_match(match):
    radiant_team = np.ndarray.tolist(np.where(X_matrix[match, 0:TOTAL_NUM_HEROES] == 1)[0])
    dire_team = np.ndarray.tolist(np.where(X_matrix[match, 0:TOTAL_NUM_HEROES:] == -1)[0])    # Syn
    for hero1, hero2, hero3 in itertools.combinations(radiant_team, 3):
        for i in itertools.permutations((hero1,hero2,hero3)):
            syn_total_matrix[i] += 1

            if y_train[match] == 1:
                syn_win_matrix[i] += 1

    for hero1, hero2, hero3 in itertools.combinations(dire_team, 3):
        for i in itertools.permutations((hero1,hero2,hero3)):
            syn_total_matrix[i] += 1

            if y_train[match] == -1:
                syn_win_matrix[i] += 1


def fill_matrix():
    # Do the synergy matrix, iterate through the top half of the matrix
    # Value of [hero i, hero j] is the same as [hero j, hero i]
    for hero1, hero2, hero3 in itertools.product(range(len(syn_matrix)), range(len(syn_matrix)), range(len(syn_matrix))):
        if hero1 == hero2 or hero2 == hero3 or hero1 == hero3:
            syn_matrix[hero1][hero2][hero3] = 0
            continue
        if syn_total_matrix[hero1, hero2, hero3] == 0:
            percentage = 0.5
        else:
            percentage = syn_win_matrix[hero1][hero2][hero3] / syn_total_matrix[hero1][hero2][hero3]

        mean_wr = (wr_matrix[hero1][0] + wr_matrix[hero2][0] + wr_matrix[hero3][0]) / 3
        syn_matrix[hero1][hero2][hero3] = percentage - mean_wr


def create_match_matrices(match):
    total_syn = []
    avg_match_syn = []
    radiant_team = np.ndarray.tolist(np.where(X_matrix[match, 0:TOTAL_NUM_HEROES] == 1)[0])
    dire_team = np.ndarray.tolist(np.where(X_matrix[match, 0:TOTAL_NUM_HEROES] == -1)[0])
    teams = [radiant_team, dire_team]

    # Get the average team synergies for each team
    for team in teams:
        for hero1, hero2, hero3 in itertools.combinations(team, 3):
            total_syn.append(syn_matrix[hero1][hero2][hero3])
        avg_match_syn.append(np.mean(total_syn))
        total_syn = []
    # Index 0 : Radiant team, Index 1: Dire team
    rad_syn_adv = avg_match_syn[0] - avg_match_syn[1]

    return rad_syn_adv


def get_wr_percentage(hero1):
    total_count, win_count = 0, 0
    for match in range(len(X_train)):
        if X_train[match][hero1] != 0:
            total_count += 1
            if y_train[match] == X_train[match][hero1]:
                win_count += 1

    return (win_count / total_count)


def fill_wr_matrix():
    for hero in tqdm(range(len(wr_matrix))):
        wr_matrix[hero][0] = get_wr_percentage(hero)


if __name__ == "__main__":
    # Create synergy and counter matricies
    syn_matrix = np.zeros((TOTAL_NUM_HEROES, TOTAL_NUM_HEROES, TOTAL_NUM_HEROES))
    syn_win_matrix = np.zeros((TOTAL_NUM_HEROES, TOTAL_NUM_HEROES, TOTAL_NUM_HEROES))
    syn_total_matrix = np.zeros((TOTAL_NUM_HEROES, TOTAL_NUM_HEROES, TOTAL_NUM_HEROES))
    wr_matrix = np.zeros((TOTAL_NUM_HEROES, 1))

    # Ask for which data file to use
    try:
        which_data = int(input("Which data file to load?\n1. Base\n2. Performance\n"))
    except ValueError:
        print("Invalid input")
        exit()
    else:
        data_name = which_data
        if which_data == 1:
            data_name = 'Base'
        elif which_data == 2:
            data_name = 'Perf'
        else:
            print("Invalid input")
            exit()

    # Load data file and define matrices
    try:
        X_matrix, y_matrix, X_train, X_test, y_train, y_test = load_data(data_name)
    except:
        print('Could not locate data, run "preprocess_data.py" first.')
        exit()
    else:
        while True:
            try:
                user_input = int(input("Lists of options to input:\n1. Fill syn and coun matrices and save them \n"
                                       "2. Load the %s dataset and append the synergy counter matrices\n3. Exit\n" % data_name))
            except ValueError:
                print("Invalid input")
                continue

            # Fill matrices
            if user_input == 1:
                # Fill the matrices
                fill_wr_matrix()
                for match in tqdm(range(len(X_train))):
                    process_match(match)
                fill_matrix()
                # Save matrices for visualisation
                np.savez(os.path.join(BASE_DIR, PROCESSED_DATA_DIR, 'WinrateMatrix'), wr_matrix=wr_matrix)
                np.savez(os.path.join(BASE_DIR, PROCESSED_DATA_DIR, '3DSynMatrx'), syn_matrix=syn_matrix)
                continue

            # Load and append matrices
            if user_input == 2:
                syn_file = np.load(os.path.join(BASE_DIR, PROCESSED_DATA_DIR, '3DSynMatrx.npz'))
                syn_matrix = syn_file['syn_matrix']

                wr_file = np.load(os.path.join(BASE_DIR, PROCESSED_DATA_DIR, 'WinrateMatrix.npz'))
                wr_matrix = wr_file['wr_matrix']

                # Add 4 extra columns to the X matrix to contain synergy and counter values and 2 for winrates
                extra_columns = np.zeros((len(X_matrix), 1))
                X_matrix = np.append(X_matrix, extra_columns, axis=1)

                # Iterate through all the matches to add these values
                for match in tqdm(range(len(X_matrix))):
                    rad_syn_adv = create_match_matrices(match)
                    X_matrix[match, -1] = rad_syn_adv

                X_train, X_test, y_train, y_test = train_test_split(X_matrix, y_matrix, test_size=0.2,
                                                                    random_state=1411)

                file_name = 'AugMatrices' if data_name == 'Base' else 'AugPerfMatrices'
                # Save the new matrices
                save_data(file_name, X_matrix, y_matrix, X_train, X_test, y_train, y_test)

            if user_input == 3:
                break
