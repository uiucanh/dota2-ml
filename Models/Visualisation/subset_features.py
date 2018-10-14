from sklearn.linear_model import LogisticRegression
from src.Models.load_data import *
from sklearn.model_selection import train_test_split
from heapq import nlargest
from functools import reduce
from src.Models.Visualisation.plot_graph import get_heroes_names
from tqdm import tqdm

import matplotlib.pyplot as plt

"""
Find test scores through subsetting the dataset and create visualisations
"""

# Get threshold value used for distinct subsets
def get_threshold(largest_list, text):
    largest_where = []

    for x in largest_list:
        largest_where.append(np.where(X_matrix[:, x] == 0))

    X_new = X_matrix[reduce(np.intersect1d, (x for x in largest_where))]
    y_new = y_matrix[reduce(np.intersect1d, (x for x in largest_where))]
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=1411)

    model.fit(X_train_new, y_train_new)
    tmp_threshold = model.score(X_test_new, y_test_new)
    print("Number of matches where " + text + " are removed: %s" % X_new.shape[0])
    print("Test accuracy is: %s" % tmp_threshold)
    print("#" * 100)
    print()

    return tmp_threshold




if __name__ == '__main__':
    np.random.seed(1411)
    model = LogisticRegression()

    X_matrix, y_matrix, X_train, X_test, y_train, y_test = load_data('Base')
    model.fit(X_train, y_train)

    highest_coefs = np.argsort(model.coef_[0])[::-1]
    heroes_list = get_heroes_names()

    # Base threshold
    threshold = model.score(X_test, y_test)
    wr_matrix = load_data('Winrate')
    print("Test accuracy of base: %s" % threshold)
    print("#" * 100)
    print()

    # Setup simluation test information
    sim_count = 1
    num_simulations = 100
    scores = []
    highest_score = [float(0)] * sim_count
    highest_subset = [[0]] * sim_count

    lowest_score = [threshold] * sim_count
    lowest_subset = [[0]] * sim_count

    # Do most popular heroes
    heroes_count = []
    for i in range(len(X_matrix[0])):
        heroes_count.append(np.count_nonzero(X_matrix[:, i]))
    # Index of 10 most popular heroes
    most_popular = nlargest(10, heroes_count)
    indicies = []
    for hero in most_popular:
        indicies.append(heroes_count.index(hero))

    threshold_popular = get_threshold(indicies, '10 most popular heroes')

    # Do heroes with largest win rates
    indicies = nlargest(10, enumerate(wr_matrix), key=lambda x: x[1])
    indicies = [x[0] for x in indicies]
    threshold_largest = get_threshold(indicies, '10 highest win rates heroes')

    # Do heroes with largest coef
    indicies = highest_coefs.tolist()[:5]
    threshold_coef = get_threshold(indicies, '10 heroes with largest coefficients')

    # Start simulation
    for j in tqdm(range(sim_count)):
        for i in range(num_simulations):
            # Get 10 random heroes to remove
            rand_heroes = np.random.randint(low=0, high=115, size=10)

            # Get the matches these 10 heroes appear in
            matches_heroes_in = []
            for x in rand_heroes:
                matches_heroes_in.append(np.where(X_matrix[:, x] == 0))

            # Remove those matches
            X_new = X_matrix[reduce(np.intersect1d, (x for x in matches_heroes_in))]
            y_new = y_matrix[reduce(np.intersect1d, (x for x in matches_heroes_in))]

            X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2,
                                                                                random_state=1411)

            # Fit and get the score of the subsetted dataset
            model.fit(X_train_new, y_train_new)
            score = model.score(X_test_new, y_test_new)
            scores.append(score)
            if score > highest_score[j]:
                highest_score[j] = score
                highest_subset[j] = rand_heroes

            if score < lowest_score[j]:
                lowest_score[j] = score
                lowest_subset[j] = rand_heroes


    plt.figure(figsize=(10, 5), dpi=100)
    plt.bar(np.arange(num_simulations * sim_count), scores)

    # Draw the values for special subsets
    plt.axhline(y=threshold_coef,linewidth=1, color='c', label='Highest Coef')
    plt.axhline(y=threshold_popular, linewidth=1, color='g', label='Most popular (10)')
    plt.axhline(y=threshold, linewidth=1, color='b', label='Base')
    plt.axhline(y=threshold_largest, linewidth=1, color='r', label='Highest win percentage')

    plt.xlabel('Simulations')
    plt.ylabel('Test accuracy')
    plt.legend()
    plt.ylim(lowest_score[0] - 0.02, highest_score[0] + 0.02, 0.01)
    plt.tight_layout()
    # plt.show()
    plt.savefig('hero_subset3.png')

    print("Highest test accuracy: %s" % highest_score)
    print("Subset:")
    print([heroes_list[i] for i in highest_subset[0]])
    print()
    print("Lowest test accuracy: %s" % lowest_score)
    print([heroes_list[i] for i in lowest_subset[0]])
    print()

    for i in range(sim_count):
        print("Highest of simulation %s: " % i)
        print(round(highest_score[i], 3), [heroes_list[j] for j in highest_subset[i]])
        print("Lowest of simulation %s: " % i)
        print(round(lowest_score[i], 3), [heroes_list[j] for j in lowest_subset[i]])
        print()
