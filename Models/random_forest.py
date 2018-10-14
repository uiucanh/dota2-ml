from src.Models.load_data import *
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
import sys


def fit_model(**kwargs):
    # Fit data
    rf = RandomForestClassifier(random_state=1411, **kwargs)
    rf.fit(X_train, y_train)

    return rf


# Prediction
def predict(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print('Accuracy of Random Forest classifier on test set: {:.2f} \n'.format(model.score(X_test, y_test)))
    print('ROC AUC score of Random Forest classifier on test set: {:.2f} \n'.format(roc_auc_score(y_test, y_pred)))
    print("Classfication report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix: \n")
    print(confusion_matrix(y_test, y_pred))


# Return cross validation score
def cross_validation(model, X_matrix, y_matrix, cv):
    kfold = model_selection.KFold(n_splits=cv)
    results = model_selection.cross_val_score(model, X_matrix, y_matrix, cv=kfold, scoring='accuracy')
    print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


# Randomised search to find the parameters that will produce the best model
def rand_search(model, X_train, y_train, cv):
    # Parameters to tune
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [10, 50, 100]
    min_samples_leaf = [10, 50, 100]

    tuned_parameters = {'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf}
    clf = RandomizedSearchCV(model, param_distributions=tuned_parameters, n_iter=5, cv=cv, scoring="accuracy")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Best parameters set found on development set:\n%s" % clf.best_params_)
    print('Accuracy of Random Forest classifier on test set: {:.2f} \n'.format(clf.score(X_test, y_test)))
    print('Mean cross-validated score: {:.2f} \n'.format(clf.best_score_))
    print('ROC AUC score of Random Forest classifier on test set: {:.2f} \n'.format(roc_auc_score(y_test, y_pred)))
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix: \n")
    print(confusion_matrix(y_test, y_pred))
    return clf


if __name__ == '__main__':
    np.random.seed(1411)
    dataset = input("What dataset to use: \n")
    number_cv = int(input("How many cv folds: \n"))
    X_matrix, y_matrix, X_train, X_test, y_train, y_test = load_data(type=dataset, scale=False)

    # Output results to txt
    text_file = os.path.join(BASE_DIR, TRAINED_MODEL_DIR, dataset + 'RFModel.txt')
    sys.stdout = open(text_file, "w")

    print("Fitting model: \n")
    # Fitting basic model
    rf = fit_model(n_estimators=100)

    predict(rf, X_test, y_test)
    cross_validation(rf, X_matrix, y_matrix, cv=number_cv)

    print()
    print("#" * 100)
    print("Performing random parameter search: \n")
    clf = rand_search(rf, X_train, y_train, cv=number_cv)

    save_model(dataset + 'RFModel', rf)
    save_model(dataset + 'GridRFModel', clf)
