# Model Evaluation and Hyperparameter Tuning

## Overview

This project evaluates multiple machine learning models on the Enron dataset and performs hyperparameter tuning for Logistic Regression and SGDClassifier. The performance of each model is assessed using accuracy and F1 score to determine the best-performing algorithm.

## Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.x
- Required Python libraries:
  - `scikit-learn`
  - `pandas`
  - `numpy`

You can install the required libraries using pip:

```bash
pip install scikit-learn pandas numpy
git clone <repository-url>
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Define the model
model = LogisticRegression()

# Set up the parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'max_iter': [1000]
}

# Conduct grid search
grid_search = GridSearchCV(model, param_grid, scoring='f1', cv=5)
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_
print("Best parameters for Logistic Regression:", best_params)
from sklearn.linear_model import SGDClassifier

# Define the model
sgd_model = SGDClassifier()

# Set up the parameter grid
sgd_param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1],
    'learning_rate': ['constant', 'optimal'],
    'max_iter': [1000],
    'loss': ['hinge', 'log']
}

# Conduct grid search
sgd_grid_search = GridSearchCV(sgd_model, sgd_param_grid, scoring='f1', cv=5)
sgd_grid_search.fit(X_train, y_train)

# Best parameters
best_sgd_params = sgd_grid_search.best_params_
print("Best parameters for SGDClassifier:", best_sgd_params)

Replace `<repository-url>` with your actual GitHub repository URL. If you need any more adjustments or additional sections, let me know!
