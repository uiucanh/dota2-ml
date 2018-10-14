import numpy as np

from src.CollectData.collect_data import setup_db_log
from src.constants import TOTAL_NUM_HEROES
from src.Models.load_data import save_data
from sklearn.model_selection import train_test_split


# Import the data from MongoDB and insert them into the dataframes
def import_from_mongo(test_size):
    X_matrix = np.zeros((TOTAL_NUM_MATCHES, TOTAL_NUM_HEROES + 5))          # Plus the 5 invalid IDs
    X_matrix_h = np.zeros((TOTAL_NUM_MATCHES, TOTAL_NUM_HEROES + 5 + 2))    # Plus 2 for performances
    X_matrix_pt = np.zeros((TOTAL_NUM_MATCHES, TOTAL_NUM_HEROES + 5 + 11))  # Plus 1 for duration, 10 for pick order
    X_matrix_a = np.zeros((TOTAL_NUM_MATCHES, TOTAL_NUM_HEROES + 5 + 13))   # Plus both duration, duration and performance
    y_matrix = np.zeros(TOTAL_NUM_MATCHES)

    # Produce the output vector
    for i, match in enumerate(find):
        if match['radiant_win']:
            y_matrix[i] = 1
        else:
            y_matrix[i] = -1

        # Insert the radiant heroes into the feature vector
        rad_heroes = list(match['radiant_team'].split(','))
        for hero in rad_heroes:
            hero_id = int(hero) - 1
            X_matrix[i, hero_id] = 1
            X_matrix_h[i, hero_id] = 1
            X_matrix_pt[i, hero_id] = 1
            X_matrix_a[i, hero_id] = 1

        # Insert the dire heroes into the feature vector
        dire_heroes = list(match['dire_team'].split(','))
        for hero in dire_heroes:
            hero_id = int(hero) - 1
            X_matrix[i, hero_id] = -1
            X_matrix_h[i, hero_id] = -1
            X_matrix_pt[i, hero_id] = -1
            X_matrix_a[i, hero_id] = -1

        # Do pick orders
        for j in range(5):
            hero_id =  fix_hero_id(match['Rad_Pick_' + str(j)])
            X_matrix_pt[i, -11 + j] = hero_id
            X_matrix_a[i, -13 + j] = hero_id

        for j in range(5):
            hero_id =  fix_hero_id(match['Dire_Pick_' + str(j)])
            X_matrix_pt[i, -6 + j] = hero_id
            X_matrix_a[i, -8 + j] = hero_id

        # # Do match performances
        X_matrix_h[i, -2] = X_matrix_a[i, -2] = match['rad_performance']
        X_matrix_h[i, -1] = X_matrix_a[i, -1] = match['dire_performance']

        X_matrix_pt[i, -1] = X_matrix_a[i, -3] = match['duration'] / 60

    # Remove empty columns
    empty_cols = [23, 114, 115, 116, 117]

    X_matrix = np.delete(X_matrix, empty_cols, axis=1)
    X_matrix_h = np.delete(X_matrix_h, empty_cols, axis=1)
    X_matrix_pt = np.delete(X_matrix_pt, empty_cols, axis=1)
    X_matrix_a = np.delete(X_matrix_a, empty_cols, axis=1)

    # Find number of differences between the two classes)
    difference = len(y_matrix[y_matrix == 1]) - len(y_matrix[y_matrix == -1])

    if difference > 0:
        list_of_indices = np.where(y_matrix == 1)
    else:
        list_of_indices = np.where(y_matrix == -1)

    np.random.seed(1411)
    np.random.shuffle(list_of_indices[0])

    # Undersample the data
    X_matrix = np.delete(X_matrix, list_of_indices[0][0:difference], axis=0)
    X_matrix_h = np.delete(X_matrix_h, list_of_indices[0][0:difference], axis=0)
    X_matrix_pt = np.delete(X_matrix_pt, list_of_indices[0][0:difference], axis=0)
    X_matrix_a = np.delete(X_matrix_a, list_of_indices[0][0:difference], axis=0)

    y_matrix = np.delete(y_matrix, list_of_indices[0][0:difference])

    # Save Base data
    X_train, X_test, y_train, y_test = train_test_split(X_matrix, y_matrix, test_size=test_size, random_state=1411)
    save_data('BaseMatrices', X_matrix, y_matrix, X_train, X_test, y_train, y_test)

    # # Save Performance data
    X_train, X_test, y_train, y_test = train_test_split(X_matrix_h, y_matrix, test_size=test_size, random_state=1411)
    save_data('PerfBaseMatrices', X_matrix_h, y_matrix, X_train, X_test, y_train, y_test)
    #
    # Save Picks + Time data
    X_train, X_test, y_train, y_test = train_test_split(X_matrix_pt, y_matrix, test_size=test_size, random_state=1411)
    save_data('TimePickBaseMatrices', X_matrix_pt, y_matrix, X_train, X_test, y_train, y_test)

    # Save All data
    X_train, X_test, y_train, y_test = train_test_split(X_matrix_a, y_matrix, test_size=test_size, random_state=1411)
    save_data('TimePickPerfBaseMatrices', X_matrix_a, y_matrix, X_train, X_test, y_train, y_test)


# Fix the hero_id to remove unused ID
def fix_hero_id(hero_id):
    if hero_id < 24:
        hero_id -= 1
    else:
        hero_id -= 2

        if hero_id == 117 or hero_id == 118:
            hero_id -= 4

    return hero_id

if __name__ == '__main__':
    db_collection, logger = setup_db_log('preprocess.log')
    test_size = 0.2

    # Ask user for input
    duration = int(input("Matches with duration less than (0 for no duration): \n"))
    if duration == 0:
        find = db_collection.find({})
    else:
        find = db_collection.find({"duration": {"$lte": duration}})

    TOTAL_NUM_MATCHES = find.count()

    import_from_mongo(test_size=test_size)
