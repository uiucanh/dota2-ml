from src.Models.load_data import *
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
import sys

def fit_model(**kwargs):
    mlp = MLPClassifier(hidden_layer_sizes=(100, ), random_state=1411, **kwargs)
    mlp.fit(X_train, y_train)

    return mlp

#Prediction
def predict(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print('Accuracy of MLP Classifier on test set: {:.2f} \n'.format(model.score(X_test, y_test)))
    print('ROC AUC score of MLP Classifier on test set: {:.2f} \n'.format(roc_auc_score(y_test, y_pred)))
    print("Classfication report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix: \n")
    print(confusion_matrix(y_test, y_pred))


# Return cross validation score
def cross_validation(model, X_matrix, y_matrix, cv):
    kfold = model_selection.KFold(n_splits=cv)
    results = model_selection.cross_val_score(model, X_matrix, y_matrix, cv=kfold, scoring='accuracy')
    print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


# Grid search to find the parameters that will produce the best model
def gridsearch(model, X_train, y_train, cv):
    tuned_parameters = {'alpha' : [0.0001, 0.001, 0.01, 0.1, 1], 'learning_rate_init': [0.001, 0.01, 0.1, 1]}
    clf = GridSearchCV(model, tuned_parameters, cv=cv, scoring="accuracy")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Best parameters set found on development set:\n%s" % clf.best_params_)
    print('Accuracy of MLP Classifier on test set: {:.2f} \n'.format(clf.score(X_test, y_test)))
    print('Mean cross-validated score: {:.2f} \n'.format(clf.best_score_))
    print(
        'ROC AUC score of MLP Classifier on test set: {:.2f} \n'.format(roc_auc_score(y_test, y_pred)))
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
    text_file = os.path.join(BASE_DIR, TRAINED_MODEL_DIR, dataset + 'MLPModel.txt')
    sys.stdout = open(text_file, "w")

    print("Fitting model: \n")
    # Fitting basic model
    mlp = fit_model(early_stopping=True)

    predict(mlp, X_test, y_test)
    cross_validation(mlp, X_matrix, y_matrix, cv=number_cv)

    print()
    print("#" * 100)
    print("Performing grid search: \n")
    clf = gridsearch(mlp, X_train, y_train, cv=number_cv)

    save_model(dataset + 'MLPModel', mlp)
    save_model(dataset + 'GridMLPModel', clf)