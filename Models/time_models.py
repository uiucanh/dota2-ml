import time
from src.Models.load_data import *

"""
Measure the time taken for each model to perform 1000 predictions
"""
models = ['Log', 'RF', 'SVC', 'MLP', 'XGB']
results = {}
X_matrix, y_matrix, X_train, X_test, y_train, y_test = load_data('Base')

for model_name in models:
    estimator = load_model('Base' + model_name)
    start = time.time()
    for i in range(1000):
        estimator.predict(X_test[i].reshape(1, -1))
    runtimes = time.time() - start
    results[model_name] = runtimes

print(results)