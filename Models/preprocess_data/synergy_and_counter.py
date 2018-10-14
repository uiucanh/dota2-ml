import itertools
from tqdm import tqdm
from src.Models.load_data import *
from src.constants import BASE_DIR, TOTAL_NUM_HEROES, PROCESSED_DATA_DIR
from sklearn.model_selection import train_test_split


# This function gets the synergy and counter percentage of hero i and hero j
# Iterate through all the recorded matches
# Check which heroes are in both teams, the hero pair in syn_total and coun_total are added by 1
# syn_win and coun_win are added by 1 depending on the result of the match
# Return percentages of win / total
def process_match(match):
    radiant_team = np.ndarray.tolist(np.where(X_train[match, 0:TOTAL_NUM_HEROES] == 1)[0])
    dire_team = np.ndarray.tolist(np.where(X_train[match, 0:TOTAL_NUM_HEROES] == -1)[0])

    # Syn
    for team in [radiant_team, dire_team]:
        for hero1, hero2 in itertools.combinations(team, 2):
            syn_total_matrix[hero1, hero2] += 1
            syn_total_matrix[hero2, hero1] += 1

            # If won
            if X_train[match][hero1] == y_train[match]:
                syn_win_matrix[hero1, hero2] += 1
                syn_win_matrix[hero2, hero1] += 1

    # Coun
    for rad_hero in radiant_team:
        for dire_hero in dire_team:
            coun_total_matrix[rad_hero, dire_hero] += 1
            coun_total_matrix[dire_hero, rad_hero] += 1

            # If won
            if X_train[match][rad_hero] == y_train[match]:
                coun_win_matrix[rad_hero, dire_hero] += 1
            else:
                coun_win_matrix[dire_hero, rad_hero] += 1


def fill_matrix():
    # Do the synergy matrix, iterate through the top half of the matrix
    # Value of [hero i, hero j] is the same as [hero j, hero i]
    for hero1 in tqdm(range(len(syn_matrix))):
        for hero2 in range(hero1 + 1, len(syn_matrix)):
            if syn_total_matrix[hero1][hero2] < 50:
                percentage = 0.5
            else:
                percentage = syn_win_matrix[hero1][hero2] / syn_total_matrix[hero1][hero2]
            # Predicted win rate = avg of individual win rates
            mean_wr = (wr_matrix[hero1][0] + wr_matrix[hero2][0]) / 2
            syn_matrix[hero1][hero2] = syn_matrix[hero2][hero1] = percentage - mean_wr
    # Do the counter matrix
    # Value of [hero j, hero i] is 1 - [hero i, hero j]
    for hero1 in tqdm(range(len(coun_matrix))):
        for hero2 in range(hero1 + 1, len(coun_matrix)):
            if coun_total_matrix[hero1][hero2] < 50:
                percentage = 0.5
            else:
                percentage = coun_win_matrix[hero1][hero2] / coun_total_matrix[hero1][hero2]

            # Get individual heroes win rates
            wr_hero1 = wr_matrix[hero1][0]
            wr_hero2 = wr_matrix[hero2][0]

            # Do predicted win rate using referenced forumla
            mean_wr = (wr_hero1 * (1 - wr_hero2)) / (wr_hero1 * (1 - wr_hero2) + wr_hero2 * (1 - wr_hero1))
            wr_against = percentage - mean_wr
            coun_matrix[hero1][hero2] = wr_against
            coun_matrix[hero2][hero1] = -wr_against


def create_match_matrices(match):
    radiant_team = np.ndarray.tolist(np.where(X_matrix[match, 0:TOTAL_NUM_HEROES] == 1)[0])
    dire_team = np.ndarray.tolist(np.where(X_matrix[match, 0:TOTAL_NUM_HEROES] == -1)[0])
    # Get the average team synergies for each team
    rad_syn_adv, rad_coun_adv, dire_syn_adv, dire_coun_adv = get_match_syncoun(radiant_team, dire_team, syn_matrix,
                                                                               coun_matrix)
    rad_avg_wr, dire_avg_wr = get_match_wr(radiant_team, dire_team, wr_matrix)
    return rad_syn_adv, rad_coun_adv, dire_syn_adv, dire_coun_adv, rad_avg_wr, dire_avg_wr


# using syn_matrix and coun_matrix, return the syn and coun values for the input match
def get_match_syncoun(radiant_team, dire_team, syn_m, coun_m):
    total_syn = []
    avg_match_syn = []
    syn_matrix = syn_m
    coun_matrix = coun_m

    # Get the syn values for both teams
    for team in [radiant_team, dire_team]:
        for hero1, hero2 in itertools.combinations(team, 2):
            total_syn.append(syn_matrix[hero1][hero2])
        avg_match_syn.append(np.mean(total_syn))
        total_syn = []
    # Index 0 : Radiant team, Index 1: Dire team
    rad_syn_adv = avg_match_syn[0]
    dire_syn_adv = avg_match_syn[1]

    total_coun = []
    # Get the average counter values for radiant team
    for rad_hero in radiant_team:
        for dire_hero in dire_team:
            total_coun.append(coun_matrix[rad_hero][dire_hero])

    rad_coun_adv = np.mean(total_coun)
    dire_coun_adv = -rad_coun_adv

    return rad_syn_adv, rad_coun_adv, dire_syn_adv, dire_coun_adv


# Using wr_matrix, return the average individual win rates for both teams
def get_match_wr(radiant_team, dire_team, wr_m):
    total_wr = []
    avg_match_wr = []
    for team in [radiant_team, dire_team]:
        for hero in team:
            total_wr.append(wr_m[hero][0])
        avg_match_wr.append(np.mean(total_wr))
        total_wr = []

    return avg_match_wr[0], avg_match_wr[1]


# Get the individual win rate of each hero
# Find matches where hero i is present and whether they won
# Get the percentage through win / total
def get_wr_percentage(hero1):
    total_count, win_count = 0, 0
    for match in range(len(X_train)):
        if X_train[match][hero1] != 0:
            total_count += 1
            if y_train[match] == X_train[match][hero1]:
                win_count += 1

    return (win_count / total_count)


# Fill the win rate matrix
def fill_wr_matrix():
    for hero in tqdm(range(len(wr_matrix))):
        wr_matrix[hero][0] = get_wr_percentage(hero)


if __name__ == "__main__":
    # Create synergy and counter matricies
    syn_matrix, coun_matrix = np.zeros((TOTAL_NUM_HEROES, TOTAL_NUM_HEROES)), np.zeros(
        (TOTAL_NUM_HEROES, TOTAL_NUM_HEROES))
    syn_win_matrix, coun_win_matrix = np.zeros((TOTAL_NUM_HEROES, TOTAL_NUM_HEROES)), np.zeros(
        (TOTAL_NUM_HEROES, TOTAL_NUM_HEROES))
    syn_total_matrix, coun_total_matrix = np.zeros((TOTAL_NUM_HEROES, TOTAL_NUM_HEROES)), np.zeros(
        (TOTAL_NUM_HEROES, TOTAL_NUM_HEROES))
    wr_matrix = np.zeros((TOTAL_NUM_HEROES, 1))

    # Ask for which data file to use
    try:
        which_data = input("Which data file to load?\n")
    except ValueError:
        print("Invalid input")
        exit()
    else:
        data_name = which_data

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
                # Save matrices
                np.savez(os.path.join(BASE_DIR, PROCESSED_DATA_DIR, SPECIAL_DATA_DIR, 'WinrateMatrices'),
                         wr_matrix=wr_matrix)
                np.savez(os.path.join(BASE_DIR, PROCESSED_DATA_DIR, SPECIAL_DATA_DIR, 'SynCounMatrices'),
                         syn_matrix=syn_matrix,
                         coun_matrix=coun_matrix)
                continue

            # Load and append matrices
            if user_input == 2:
                syn_file = np.load(os.path.join(BASE_DIR, PROCESSED_DATA_DIR, SPECIAL_DATA_DIR, 'SynCounMatrices.npz'))
                syn_matrix, coun_matrix = syn_file['syn_matrix'], syn_file['coun_matrix']

                wr_file = np.load(os.path.join(BASE_DIR, PROCESSED_DATA_DIR, SPECIAL_DATA_DIR, 'WinrateMatrices.npz'))
                wr_matrix = wr_file['wr_matrix']

                # Add 4 extra columns to the X matrix to contain synergy and counter values and 2 for winrates
                extra_columns = np.zeros((len(X_matrix), 3))
                X_matrix = np.append(X_matrix, extra_columns, axis=1)

                # Iterate through all the matches to add these values
                for match in tqdm(range(len(X_matrix))):
                    rad_syn_adv, rad_coun_adv, dire_syn_adv, dire_coun_adv, rad_avg_wr, dire_avg_wr = create_match_matrices(
                        match)
                    X_matrix[match, -3] = rad_syn_adv - dire_syn_adv
                    X_matrix[match, -2] = rad_coun_adv
                    X_matrix[match, -1] = rad_avg_wr - dire_avg_wr

                X_train, X_test, y_train, y_test = train_test_split(X_matrix, y_matrix, test_size=0.2,
                                                                    random_state=1411)
                if data_name == 'Base':
                    file_name = 'AugMatrices'
                elif data_name == 'PerfBase':
                    file_name = 'PerfAugMatrices'
                else:
                    file_name = data_name + 'Aug' + 'Matrices'
                # Save the new matrices
                save_data(file_name, X_matrix, y_matrix, X_train, X_test, y_train, y_test)

            if user_input == 3:
                exit()
