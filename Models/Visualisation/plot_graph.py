import seaborn as sns
import matplotlib.pyplot as plt
import json
import plotly.plotly as py
import plotly.graph_objs as go

from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from src.Models.load_data import *
from sklearn.model_selection import learning_curve
from src.constants import BASE_DIR, TOTAL_NUM_HEROES


def draw_heroes_hist(save=False):
    X_matrix, y_matrix, X_train, X_test, y_train, y_test = load_data('Base')
    # A list of the total amount of matches for hero ID i which correspond to the index of this list
    heroes_count = []
    for i in range(TOTAL_NUM_HEROES):
        heroes_count.append(np.count_nonzero(X_matrix[:, i]) / X_matrix.shape[0])

    # Setup the x-axis values
    index = np.arange(len(heroes_list))

    plt.figure(figsize=(20, 30), dpi=100)
    plt.barh(index, heroes_count, height=0.5)
    plt.ylabel('Heroes', fontsize=30, labelpad=20)
    plt.xlabel('Percentage', fontsize=30, labelpad=20)
    plt.yticks(index, heroes_list, fontsize=12, rotation='horizontal', style='italic')
    plt.margins(y=0.01)
    plt.title('Percentage of matches where a hero is present', fontsize=50)
    ax = plt.gca()
    # Moidfy the title
    ax.title.set_position([.5, 1.01])

    # Make the count values to show next to the bars
    for i, v in enumerate(heroes_count):
        percent = round(v * 100, 2)
        ax.text(v + 0.01, i - 0.3, str(percent), color='blue')
    if not save:
        plt.show()
    else:
        plt.savefig('hero_num_matches.png')

def draw_wr_hist(save=False):
    # Load the winrate matrix
    wr_matrix = load_data('Winrate')
    winrate_count = [round(rate[0] * 100, 2) for rate in wr_matrix]

    # Setup the x-axis values
    index = np.arange(len(heroes_list))

    plt.figure(figsize=(20, 30), dpi=100)
    plt.barh(index, winrate_count, height=0.5, color='peru')
    plt.ylabel('Heroes', fontsize=30, labelpad=20)
    plt.xlabel('Win Rate', fontsize=30, labelpad=20)
    plt.xlim(35,60, 1)
    plt.yticks(index, heroes_list, fontsize=12, rotation='horizontal', style='italic')
    plt.margins(y=0.01)
    plt.title('Win Rate of Each Hero', fontsize=50)
    ax = plt.gca()
    # Moidfy the title
    ax.title.set_position([.5, 1.01])

    # Make the count values to show next to the bars
    for i, v in enumerate(winrate_count):
        ax.text(v + 0.5, i - 0.3, str(v), color='darkred')
    if not save:
        plt.show()
    else:
        plt.savefig('hero_win_rate.png')


# Get the names of all the heroes and put into a list
def get_heroes_names():
    tmp_list = []
    with open(os.path.join(BASE_DIR, 'Models/heroes.json')) as f:
        json_file = json.load(f)
    for json_entry in json_file:
        tmp_list.append(json_entry['localized_name'])
    return tmp_list


# Draw heat maps of synergy and counter matrices
def draw_heatmap(save=False):
    syn_matrix, coun_matrix = load_data('SynCoun')
    matrices = {'syn_matrix': syn_matrix, 'coun_matrix': coun_matrix}

    # Draw syn matrix
    for key in matrices:
        annotation = np.round(matrices[key], 2)
        sns.set(font_scale=2)
        fig, ax = plt.subplots(figsize=(60, 60))

        # Set pallette for colorbar
        if key == 'syn_matrix':
            cmap = "YlGnBu"
        else:
            cmap = "OrRd"
        ax = sns.heatmap(matrices[key], square=True,
                         linewidths=.1, cmap=cmap, cbar_kws={"shrink": .80},
                         xticklabels=heroes_list, yticklabels=heroes_list, annot_kws={"size": 8})
        plt.xlabel('Hero 2', fontsize=150, labelpad=40)
        plt.ylabel('Hero 1', fontsize=150, labelpad=40)

        # Modify colorbar labels
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=50)
        if not save:
            plt.show()
        else:
            plt.savefig(key + '.png')


# Draw the two heatmaps on plotly
def web_heatmap():
    syn_matrix, coun_matrix = load_data('SynCoun')
    matrices = {'syn_matrix': syn_matrix, 'coun_matrix': coun_matrix}

    for key in matrices:
        if key == 'syn_matrix':
            cmap = "YlGnBu"
            title = 'Heatmap of synergistic relationship between heroes'
        else:
            cmap = "YlOrRd"
            title = 'Heatmap of antagonistic relationship of Hero 1 (y) against Hero 2 (x)'

        trace = [go.Heatmap(z=matrices[key], x=heroes_list, y=heroes_list, colorscale=cmap)]
        layout = go.Layout(title=title, xaxis=dict(title='Hero 2'),
                           yaxis=dict(title='Hero 1', autorange='reversed'))
        fig = go.Figure(data=trace, layout=layout)

        py.iplot(fig, filename=key + "_heatmap")


# Plot the learning curve for the input models
def draw_lc_cv(model, model_name, cv, X_matrix, y_matrix, ylim=(0.58, 0.67, 0.01),
                        train_sizes=np.linspace(.1, 1.0, 10), save=False):
    plt.figure(figsize=(6, 5), dpi=200)
    sns.set_style("whitegrid")

    plt.title("Learning curve of %s" % model_name, fontsize=11)
    plt.xlabel("Number of training samples", fontsize=11, labelpad=10)
    plt.ylabel("Score", fontsize=11, labelpad=5)
    # If the model is hypertuned
    if 'Tuned' in model_name:
        train_sizes, train_scores, test_scores = learning_curve(
            model.best_estimator_, X_matrix, y_matrix, cv=cv, train_sizes=train_sizes)
    else:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_matrix, y_matrix, cv=cv, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    # Add visualisation of variances
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.ylim(min(test_scores_mean) - 0.03, max(train_scores_mean) + 0.03, 0.01)
    ax = plt.gca()
    # Moidfy the title
    ax.title.set_position([.5, 1.05])
    plt.legend(loc="best")
    if not save:
        plt.show()
    else:
        plt.savefig(model_name + ' LC.png')


# A plot of test accuracy vs number of trees in RF
def draw_rf_trees():
    X_matrix, y_matrix, X_train, X_test, y_train, y_test = load_data('Base')
    number_trees = np.linspace(100, 1000, 10, dtype=int)
    results = []

    for tree in tqdm(number_trees):
        rf = RandomForestClassifier(random_state=1411, n_estimators=tree, warm_start=True)
        rf.fit(X_train, y_train)
        results.append(rf.score(X_test, y_test))

    plt.plot(number_trees, results)
    plt.xlabel('Number of trees')
    plt.ylabel('Test Accuracy')
    plt.savefig('RF tree.png')


# Plotting features importances for RF or XGB
def forest_features_importance(model, dataset, X_matrix, which_model, max_features):
    sns.set(style='whitegrid')
    importances = model.feature_importances_
    if which_model == 'RF':
        # Get the standard deviation of these feature importances in all of the trees
        std = np.std([tree.feature_importances_ for tree in model.estimators_],
                     axis=0)
    indices = np.argsort(importances)[::-1]

    # Add new labels for x-axis
    if dataset == 'A':
        labels = heroes_list
    elif dataset == 'B':
        labels = heroes_list + ['rad_syn_ddv', 'rad_coun_adv', 'rad_wr_adv']
    elif dataset == 'C':
        labels = heroes_list + ['rad_avg_perf', 'dire_avg_perf']
    elif dataset == 'D':
        labels = heroes_list + ['rad_time_wr', 'dire_time_wr', 'rad_pick_wr', 'dire_pick_wr']
    else:
        labels = heroes_list + ['rad_avg_perf', 'dire_avg_perf', 'rad_time_wr', 'dire_time_wr', 'rad_pick_wr', 'dire_pick_wr', 'rad_syn_ddv', 'rad_coun_adv', 'rad_wr_adv']
    tick_labels = []
    for i in range(len(indices)):
        tick_labels.append(labels[indices[i]])

    plt.figure(figsize=(10, 8), dpi=200)
    if which_model == 'RF':
        plt.bar(range(X_matrix.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
    else:
        plt.bar(range(X_matrix.shape[1]), importances[indices],
                color="r", align="center")
    plt.yticks(fontsize=15)
    plt.xticks(range(X_matrix.shape[1]), tick_labels, rotation='vertical', fontsize=15)
    plt.xlim([-0.5, int(max_features) + 0.5])
    plt.tight_layout()
    plt.savefig('%s Features Importances on Dataset %s.png' % (str(which_model), str(dataset)))


if __name__ == '__main__':
    list_of_options = [i for i in range(1,10)]
    heroes_list = get_heroes_names()

    while True:
        try:
            user_input = int(input("Which plot to graph?\n1. Number of matches for each hero"
                                   "\n2. Heatmap (offline)\n3. Heatmap (online)"
                                   "\n4. Learning curve vs number of samples(training vs validation)"
                                   "\n5. Learning curve vs number of samples (training vs testing)"
                                   "\n6. Random Forest number of trees learning curve"
                                   "\n7. Winrate graph"
                                   "\n8. Feature Importances of a RF model"
                                   "\n9. Test accuracy vs Number of trees in BaseRF\n"))
        except ValueError:
            print("Invalid input")
            continue
        else:
            if user_input not in list_of_options:
                print("Invalid input")
            elif user_input == 1:
                draw_heroes_hist(save=True)
            elif user_input == 2:
                draw_heatmap(save=True)
            elif user_input == 3:
                web_heatmap()
            elif user_input == 4:
                which_data = input("Which data to load?\n")
                X_matrix, y_matrix, X_train, X_test, y_train, y_test = load_data(which_data)
                which_model = input("Which model to load?\n")
                model = load_model(which_data + which_model)
                graph_name = input("Enter model name\n")
                draw_lc_cv(model, graph_name,
                                    10, X_matrix, y_matrix, save=True)
            elif user_input == 5:
                pass
            elif user_input == 6:
                pass
            elif user_input == 7:
                draw_wr_hist(save=True)
            elif user_input == 8:
                which_data = input("Which data to load?\n")
                X_matrix, y_matrix, X_train, X_test, y_train, y_test = load_data(which_data)
                which_model = input("XGB or RF?\n")
                model = load_model(which_data + which_model)
                which_dataset = input("Which dataset is that?\n")
                max_features = input("Input maximum number of features to plot. Enter 0 for default.\n")
                if int(max_features) == 0:
                    max_features = 10
                forest_features_importance(model, which_dataset, X_matrix, which_model, max_features)
            elif user_input == 9:
                draw_rf_trees()
