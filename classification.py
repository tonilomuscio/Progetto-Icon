import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from logical_classes import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, \
    recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.naive_bayes import CategoricalNB


def construct_confusion_matrix(matrix, display_labels):
    # display the confusion matrix constructed through confusion_matrix function
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=display_labels)
    disp.plot(cmap=plt.cm.YlGnBu)

    # Remove cell gridlines
    plt.grid(which='major')

    # Adjust the size of the plot
    plt.gcf().set_size_inches(10, 7)

    # Show the plot
    plt.show()


def classification(dataset_path) -> list[ClassificationResult]:
    # executes classification task for every model implemented
    classification_results = []

    warnings.filterwarnings('ignore')

    pd.set_option('display.max_columns', None)

    def train_evaluate_model(test_set, predictions, prediction_proba, model_name):
        # calculates metrics for the evaluation of the classification model
        accuracy = accuracy_score(test_set, predictions)
        f1 = f1_score(test_set, predictions, average='micro')
        precision = precision_score(test_set, predictions, average='micro')
        recall = recall_score(test_set, predictions, average='micro')
        balanced_accuracy = balanced_accuracy_score(test_set, predictions)
        if dataset_path != './dataset/Iris.csv':
            auc = roc_auc_score(test_set, prediction_proba)
        else:
            auc = roc_auc_score(test_set, prediction_proba, multi_class='ovr')

        classification_results.append(
            ClassificationResult(model_name, accuracy, f1, precision, recall,
                                 balanced_accuracy, auc, None, dataset_path)
        )

        evaluated_dataframe = pd.DataFrame(
            [[accuracy, f1, precision, recall, balanced_accuracy, auc]],
            columns=['accuracy', 'f1_score', 'precision', 'recall', 'balanced_accuracy', 'auc']
        )

        return evaluated_dataframe

    print("\n\n=========================================== PROJECT DATA ==============================================")

    raw_file = pd.read_csv(dataset_path)
    print(raw_file.info())

    dataset = raw_file[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]]

    y = (dataset['Species'])  # Assign to y only the output feature 'Species'
    x = dataset.loc[:, dataset.columns != 'Species']  # Assign to X the input features, so everything but 'Species'
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    # -----------------
    # K Nearest Neighbor
    # -----------------
    print('\n----------------------------------------- Now training with KNN -----------------------------------------')

    # Define parameter range
    knn_param_grid = {
        'n_neighbors': range(1, 10),
        'weights': ['distance', 'uniform']
    }

    knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, refit=True)

    # Fit the model for grid search function
    knn_grid.fit(x_train, y_train)

    # Print the best parameters
    print('\nKNN grid best params:\n', knn_grid.best_params_)

    # Predict with the best parameter
    knn_target_test_prd = knn_grid.predict(x_test)
    if dataset_path != './dataset/Iris.csv':
        knn_target_test_prd_proba = knn_grid.predict_proba(x_test)[:, 1]
    else:
        knn_target_test_prd_proba = knn_grid.predict_proba(x_test)

    # Construct confusion matrix
    knn_conf_matrix = confusion_matrix(y_test, knn_target_test_prd, labels=knn_grid.classes_)
    construct_confusion_matrix(knn_conf_matrix, knn_grid.classes_)

    # -----------------
    # Decision Tree
    # -----------------
    print('\n----------------------------------------- Now training with DT ------------------------------------------')

    dt = DecisionTreeClassifier(random_state=42)
    dt = dt.fit(x_train, y_train)

    # Define parameter range
    dt_param_grid = {
        'max_depth': range(1, dt.tree_.max_depth + 1, 2),
        'max_features': range(1, len(dt.feature_importances_) + 1)
    }
    dt_grid = GridSearchCV(dt, dt_param_grid, n_jobs=-1)

    # Fit the model for grid search function
    dt_grid.fit(x_train, y_train)

    # Print the best parameters
    print('\nDT grid best params:\n', dt_grid.best_params_)

    # Predict with the best parameter
    dt_target_test_prd = dt_grid.predict(x_test)
    if dataset_path != './dataset/Iris.csv':
        dt_target_test_prd_proba = dt_grid.predict_proba(x_test)[:, 1]
    else:
        dt_target_test_prd_proba = dt_grid.predict_proba(x_test)

    # Construct confusion matrix
    dt_conf_matrix = confusion_matrix(y_test, dt_target_test_prd, labels=dt_grid.classes_)
    construct_confusion_matrix(dt_conf_matrix, dt_grid.classes_)

    # -----------------
    # Random Forest
    # -----------------
    print('\n----------------------------------------- Now training with RF ------------------------------------------')

    rf = RandomForestClassifier(
        oob_score=True,
        random_state=42,
        warm_start=True,
        n_jobs=-1
    )

    # Define parameter range
    rf_param_grid = {
        'n_estimators': [15, 20, 30, 40],
        'max_depth': range(1, 10),
    }
    rf_grid = GridSearchCV(rf, rf_param_grid)

    # Fit the model for grid search
    rf_grid.fit(x_train, y_train)

    # Print the best parameters
    print('\nRF grid best params:\n', rf_grid.best_params_)

    # Predict with the best parameter
    rf_target_test_prd = rf_grid.predict(x_test)
    if dataset_path != './dataset/Iris.csv':
        rf_target_test_prd_proba = rf_grid.predict_proba(x_test)[:, 1]
    else:
        rf_target_test_prd_proba = rf_grid.predict_proba(x_test)

    # Construct confusion matrix
    rf_conf_matrix = confusion_matrix(y_test, rf_target_test_prd, labels=rf_grid.classes_)
    construct_confusion_matrix(rf_conf_matrix, rf_grid.classes_)

    # -----------------
    # Naive Bayes
    # -----------------
    print('\n----------------------------------------- Now training with NB ------------------------------------------')

    nb = CategoricalNB()

    # Model training
    nb.fit(x_train, y_train)

    # Predict with the best parameter
    nb_target_test_prd = nb.predict(x_test)
    if dataset_path != './dataset/Iris.csv':
        nb_target_test_prd_proba = nb.predict_proba(x_test)[:, 1]
    else:
        nb_target_test_prd_proba = nb.predict_proba(x_test)

    print('\nNo best params needed')

    # Construct confusion matrix
    nb_conf_matrix = confusion_matrix(y_test, nb_target_test_prd, labels=nb.classes_)
    construct_confusion_matrix(nb_conf_matrix, nb.classes_)

    # -----------------
    # Neural Network
    # -----------------
    print('\n----------------------------------------- Now training with NN ------------------------------------------')

    sc = MinMaxScaler()
    scaler = sc.fit(x_train)

    # utilizing the MinMaxScaler we normalize the range of values in our data
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # define parameter_grid
    param_grid = {
        'hidden_layer_sizes': (4, 4, 4),
        'max_iter': [2000, 2100],
        'activation': ['relu'],
        'solver': ['lbfgs'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }

    mlp_clf = MLPClassifier()

    nn_grid = GridSearchCV(mlp_clf, param_grid, n_jobs=-1)

    # fit the model for grid search
    nn_grid.fit(x_train_scaled, y_train)

    # Print the best parameters
    print('\nNN grid best params:\n', nn_grid.best_params_)

    # Predict with the best parameter
    nn_target_test_prd = nn_grid.predict(x_test_scaled)
    if dataset_path != './dataset/Iris.csv':
        nn_target_test_prd_proba = nn_grid.predict_proba(x_test_scaled)[:, 1]
    else:
        nn_target_test_prd_proba = nn_grid.predict_proba(x_test_scaled)

    # Construct confusion matrix
    nn_conf_matrix = confusion_matrix(y_test, nn_target_test_prd, labels=nn_grid.classes_)
    construct_confusion_matrix(nn_conf_matrix, nn_grid.classes_)

    # -----------------
    # Logistic Regression
    # -----------------
    print('\n--------------------------------------- Building a LR classifier ----------------------------------------')

    sc = StandardScaler()
    scaler = sc.fit(x_train)

    # utilizing the StandardScaler we normalize the range of values in our data
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Define parameter grid
    parameters = {
        'penalty': ['l2'],
        'C': np.logspace(-3, 3, 7),
        'solver': ['newton-cg', 'lbfgs', 'liblinear'],
    }

    logreg = LogisticRegression()
    lr_grid = GridSearchCV(logreg, param_grid=parameters, cv=5)

    # Fit the model for grid search
    lr_grid.fit(x_train_scaled, y_train)

    # Print the best parameters
    print('\nLR grid best params:\n', lr_grid.best_params_)

    # Predict with the best parameter
    lr_target_test_prd = lr_grid.predict(x_test_scaled)
    if dataset_path != './dataset/Iris.csv':
        lr_target_test_prd_proba = lr_grid.predict_proba(x_test_scaled)[:, 1]
    else:
        lr_target_test_prd_proba = lr_grid.predict_proba(x_test_scaled)

    # Construct confusion matrix
    lr_conf_matrix = confusion_matrix(y_test, lr_target_test_prd, labels=lr_grid.classes_)
    construct_confusion_matrix(lr_conf_matrix, lr_grid.classes_)

    # -----------------
    # K-Means
    # -----------------
    print('\n---------------------------------------- Now predict with KMeans ----------------------------------------')

    sc = MinMaxScaler()
    scaler = sc.fit(x_train)

    # utilizing the MinMaxScaler we normalize the range of values in our data
    x_scaled = scaler.transform(x)

    scaled_dataframe = pd.DataFrame(x_scaled, columns=x.columns)

    if dataset_path == './dataset/Iris.csv':
        n_cluster = 3
    else:
        n_cluster = 2

    kmeans_model = KMeans(n_cluster, n_init=10)

    # training the model
    kmeans_model.fit(scaled_dataframe)

    scaled_dataframe['cluster'] = kmeans_model.labels_

    classification_results.append(
        ClassificationResult('k_means', None, None, None, None, None, None,
                             scaled_dataframe['cluster'].value_counts(ascending=True), dataset_path)
    )

    # Print classification results
    print('\n', scaled_dataframe['cluster'].value_counts(sort=True))

    print('\n----------------------------------------------- Results ------------------------------------------------')

    knn_results = train_evaluate_model(y_test, knn_target_test_prd, knn_target_test_prd_proba, 'knn')
    knn_results.index = ['K Nearest Neighbors']

    dt_results = train_evaluate_model(y_test, dt_target_test_prd, dt_target_test_prd_proba, 'decision_trees')
    dt_results.index = ['Decision Trees']

    rf_results = train_evaluate_model(y_test, rf_target_test_prd, rf_target_test_prd_proba, 'random_forest')
    rf_results.index = ['Random Forest']

    nb_results = train_evaluate_model(y_test, nb_target_test_prd, nb_target_test_prd_proba, 'naive_bayes')
    nb_results.index = ['Naive Bayes']

    nn_results = train_evaluate_model(y_test, nn_target_test_prd, nn_target_test_prd_proba, 'neural_network')
    nn_results.index = ['Neural Network']

    lr_results = train_evaluate_model(y_test, lr_target_test_prd, lr_target_test_prd_proba, 'logistic_regression')
    lr_results.index = ['Logistic Regression']

    # list of all the created dataframes
    frames = [knn_results, dt_results, rf_results, nb_results, nn_results, lr_results]

    # transform list in a single table
    results = pd.concat(frames)

    display(results)

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    # create window to display the table containing the classification results
    the_table = ax.table(rowLabels=results.index, cellText=results.values, colLabels=results.columns, loc='center')
    the_table.auto_set_font_size(False)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    the_table.set_fontsize(10)

    # display the new table
    plt.show()

    return classification_results
