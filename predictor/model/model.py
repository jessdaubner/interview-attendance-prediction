"""Trains and evaluates model. Writes trained model's artifact to MODEL_DIR."""
import os
import argparse
import pandas as pd
from datetime import datetime
from sklearn.externals import joblib

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, \
    accuracy_score, roc_curve, average_precision_score, auc

from predictor.config import logger, LABELED_DATA, LABEL, \
    MODEL_DIR, FEATURES, UNLABELED_DATA, PREDICTIONS


def impute_missing_values(df):
    """Fill null feature values with modal values
    Args:
       df (pd.DataFrame): feature data with Nan values
    Returns
       pd.DataFrame with imputed valuees
    """
    for column in FEATURES:
        mode = df[column].mode()[0]
        df[column] = df[column].fillna(mode)
        logger.info(f'Filled NAs of column {column} with {mode}')
    return df


def transform(df):
    """Transform categorical features into numerical representations."""
    return pd.get_dummies(df, columns=FEATURES)


def preprocess(DATA):
    """Loads CSV data and executes various preprocessing including
       imputing missing values and
    Args:
       DATA (str): path to cleaned CSV data file
    Returns:
       pd.DataFrame ready to be used for model training, evaluation or inference
    """
    df = pd.read_csv(DATA)
    df = df.set_index('candidate_id')
    df = impute_missing_values(df)
    return df


def create_train_test_datasets(df, test_size):
    """Splits labeled dataset in train and test sets
    Args:
      df (pd.DataFrame):
      test_size (float): percentage of labeled data to be used in test set
    Returns:
      pd.DataFrames of train and test feature and target data (i.e.
      x_train, x_test, y_train, y_test)
    """
    X = df[FEATURES]
    X = transform(X)
    Y = df[LABEL].map({'YES': 1, 'NO': 0})
    logger.info('Splitting labeled data into train and test set with {} split'.
                format(str(int((1 - test_size)) * 100) + '/' + str(int(test_size * 100))))
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,
                                                        stratify=Y,
                                                        random_state=42)
    logger.info('Training set created with {} rows and {} features'
                .format(x_train.shape[0], x_train.shape[1]))
    logger.info('Test set created with {} rows and {} features'
                .format(x_test.shape[0], x_test.shape[1]))
    return x_train, x_test, y_train, y_test


def tune_hyperparameters(x_train, x_test, y_train, y_test):
    """Use cross-validation and grid search to find optimal
       hyper-parameters to find the best parameters on
       the training set.
    """
    parameters = [{'kernel': ['rbf'],
                   'gamma': [1e-3, 1e-4],
                   'C': [1, 10, 100, 1000],
                   'tol': [1e-2, 1e-3, 1e-4]},
                  {'kernel': ['linear'],
                   'C': [1, 10, 100, 1000],
                   'tol': [1e-2, 1e-3, 1e-4]}]

    for metric in ['precision', 'recall']:
        logger.info(f"Tuning hyper-parameters for {metric}")
        clf = GridSearchCV(SVC(), parameters, cv=6, scoring=f'{metric}_macro')
        clf.fit(x_train, y_train)

        best_params = clf.best_params_
        logger.info(f"Best parameters:")
        for metric, value in best_params.items():
            logger.info(f"{metric}: {value}")

        logger.info("Grid scores on training set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            logger.info(f"{mean:.3} (+/-%{std*2:.3}) for {params}")
        logger.info("Classification report:")
        y_true, y_pred = y_test, clf.predict(x_test)
        logger.info(classification_report(y_true, y_pred))
        return best_params


def calculate_metrics(y_pred, y_true):
    """Calculate performance metrics"""
    log_msg = f'Test Set Evaluation Metrics'
    logger.info(len(log_msg) * '-')
    logger.info(log_msg)
    logger.info(len(log_msg) * '-')

    accuracy = accuracy_score(y_true, y_pred)
    logger.info(f'Accuracy: {accuracy}')
    fpr, tpr, thresholds = roc_curve(y_true.values, y_pred, pos_label=1)
    auc_score = auc(fpr, tpr)
    logger.info(f'AUC: {auc_score:3}')
    avg_precision = average_precision_score(y_true, y_pred)
    logger.info(f'Average Precision: {avg_precision:3}')


def train_and_evaluate(cross_validate):
    """Train and evaluate model
    Args:
      cross_validate (boolean): execute hyper-parameter tuning via grid search
    Returns:
      trained pipeline
    """
    labeled_data = preprocess(LABELED_DATA)
    x_train, x_test, y_train, y_test = create_train_test_datasets(labeled_data, test_size=0.2)

    if cross_validate:
        params = tune_hyperparameters(x_train, x_test, y_train, y_test)
    else:
        # Parameters learned from previous cross-validation run
        params = {'C': 10, 'gamma': 0.001, 'kernel': 'rbf', 'tol': 0.01}

    pipeline = Pipeline([
        ('feature_interactions', PolynomialFeatures(degree=2)),
        ('classifier', SVC(kernel=params['kernel'],
                           gamma=params['gamma'],
                           C=params['C'],
                           tol=params['tol'],
                           probability=True))
    ])

    logger.info('Fiting pipeline on training data set.........')
    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    calculate_metrics(predicted, y_test)

    training_columns = x_train.columns
    return pipeline, training_columns


def save_model(pipeline, model_type='svm',
               date=datetime.now().strftime("%m-%d-%Y")):
    model_name = f'{model_type}-model-{date}.joblib'
    logger.info(f'Writing {model_name} to {MODEL_DIR}')
    joblib.dump(pipeline, os.path.join(MODEL_DIR, model_name), compress=1)


def make_predictions(pipeline, feature_cols):
    """Make predictions on unlabeled data using the
       fitted pipeline and write predictions to CSV file.
    Args:
      pipeline (sklearn.): pipeline containing fitted model and transformations
      feature_cols (list): features present in training data used to train model
    Returns:
      None
    """
    logger.info('Making predictions on unlabeled data.........')
    unlabeled_data = preprocess(UNLABELED_DATA)
    unlabeled_data = unlabeled_data[FEATURES]
    unlabeled_data = transform(unlabeled_data)
    unlabeled_data = unlabeled_data.reindex(columns=feature_cols, fill_value=0)

    predicted_attendance = pipeline.predict(unlabeled_data)
    attendance_prob = pipeline.predict_proba(unlabeled_data)
    predictions = pd.DataFrame({'Name(Cand ID)': unlabeled_data.index.values,
                                'Predicted Attendance': predicted_attendance,
                                'Probability of Attendance': attendance_prob[:, 1]})
    predictions['Predicted Attendance'] = predictions['Predicted Attendance'].map({1: 'YES', 0: 'NO'})
    predictions.to_csv(path_or_buf=PREDICTIONS, index=False)
    n_pred = predictions.shape[0]
    logger.info(f'Wrote predictions for {n_pred} candidates to {PREDICTIONS}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', action='store_true',
                        help='Specify value only if hyper-parameter tuning should be executed')

    arg = parser.parse_args()
    pipeline, feature_columns = train_and_evaluate(cross_validate=arg.tune)
    save_model(pipeline)
    make_predictions(pipeline, feature_columns)
