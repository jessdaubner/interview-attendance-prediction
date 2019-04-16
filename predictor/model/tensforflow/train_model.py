"""Trains and evaluates model. Writes trained model's artifact to MODEL_DIR."""
import shutil
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.feature_column import categorical_column_with_vocabulary_list, \
    categorical_column_with_hash_bucket, crossed_column

from predictor.preprocess.vocabulary import CLEANED_VALUES
from predictor.config import logger, LABELED_DATA, TRAIN_DATA, TEST_DATA, \
    MODEL_DIR, CSV_COLUMNS, LABEL_COLUMN, UNLABELED_DATA


def create_train_test_datasets(test_size):
    """Creates train and test datasets from labeled CSV data
       and creates two new train and test CSV files.
    Args:
      test_size (float): percentage of labeled data to be used in test set
    Returns:
      None
    """
    labeled_data = pd.read_csv(LABELED_DATA)
    train, test = train_test_split(labeled_data,
                                   test_size=test_size,
                                   random_state=42,
                                   stratify=labeled_data['attendance'])
    logger.info(f'Writing train set to {TRAIN_DATA}')
    train.to_csv(path_or_buf=TRAIN_DATA, index=False)
    logger.info(f'Writing train set to {TEST_DATA}')
    test.to_csv(path_or_buf=TEST_DATA, index=False)


def impute_missing_values(df):
    """Fill null feature values with modal values
    Args:
      df (pd.DataFrame)
    Returns
      pd.DataFrame with imputed valuees
    """
    for column in CSV_COLUMNS[1:-1]:  # Skip candidate_id and attendance
        mode = df[column].mode()[0]
        logger.debug(f'Filling NAs of column {column} with {mode}')
        df[column] = df[column].fillna(mode)
    return df


def input_fn(filename, num_epochs=None, shuffle=True):
    """Defines features and labels in training dataset to be passed to estimator"""
    df = pd.read_csv(filename,
                     names=CSV_COLUMNS[1:],
                     usecols=CSV_COLUMNS[1:], # Skip candidate_id
                     skiprows=1)
    df = impute_missing_values(df)
    labels = df['attendance'].apply(lambda x: x == 'YES').astype(int)
    return tf.estimator.inputs.pandas_input_fn(
        x=df,
        y=labels,
        batch_size=10,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=4,
    )


# See https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py
def serving_input_fn(filename):
    """For serving predictions on unlabeled data"""
    feature_placeholders = {
        column_name: tf.placeholder(tf.string, [None]) for column_name in CSV_COLUMNS[1:-1]
    }
    features = feature_placeholders.copy()
    # TODO: Custom output like https://www.tensorflow.org/guide/custom_estimators#predict needed:
    return tf.estimator.export.ServingInputReceiver(features, # dict of string to tensor specifying features to be passed to model
                                                    feature_placeholders # receiver tensors specifying input nodes where this receiver expects to be fed by default.)


def add_eval_metrics(labels, predictions):
    """Calculates AUC using 'caretful_interpolation' instead of
       the default trapeziodal method, which generates a
       WARN message: 'Trapezoidal rule is known to produce incorrect PR-AUCs'.
    """
    return {
        'auc_precision_recall_ci': tf.metrics.auc(
            labels=labels, predictions=predictions['logistic'], num_thresholds=200,
            curve='PR', summation_method='careful_interpolation')
    }


def build_estimator(model_dir):
    """Transforms and engineers features and creates estimator model."""
    vocab_features = [
        categorical_column_with_vocabulary_list(name, vocabulary_list=CLEANED_VALUES[name])
        for name, vocab_list in CLEANED_VALUES.items()
        if vocab_list and name != LABEL_COLUMN
    ]

    hash_bucket_features = [
        categorical_column_with_hash_bucket('position_skillset', hash_bucket_size=28),
        categorical_column_with_hash_bucket('candidate_skillset', hash_bucket_size=10),
        categorical_column_with_hash_bucket('interview_venue', hash_bucket_size=10),
        categorical_column_with_hash_bucket('candidate_current_location', hash_bucket_size=10),
        categorical_column_with_hash_bucket('candidate_native_location', hash_bucket_size=20),
        categorical_column_with_hash_bucket('candidate_job_location', hash_bucket_size=10)
    ]

    crossed_features = [
        crossed_column(['position_skillset', 'candidate_skillset'], hash_bucket_size=1000),
        crossed_column(['interview_venue', 'candidate_current_location', 'candidate_job_location', 'candidate_native_location'],
                       hash_bucket_size=1000),
        crossed_column(['candidate_job_location', 'candidate_current_location'], hash_bucket_size=1000),
    ]

    features = vocab_features + hash_bucket_features + crossed_features

    estimator = tf.estimator.LinearClassifier(model_dir=MODEL_DIR, feature_columns=features)
    estimator = tf.contrib.estimator.add_metrics(estimator, add_eval_metrics)
    return estimator


def log_metrics(results, data_type):
    data_type = data_type.split('_')[0].capitalize()
    log_msg = f'{data_type} Set Evaluation Metrics'
    logger.info(len(log_msg) * '-')
    logger.info(log_msg)
    logger.info(len(log_msg) * '-')
    for key in sorted(results):
        logger.info("{}: {}".format(key.upper(), results[key]))


def train_and_evaluate(estimator, steps=1000):
    """Trains and evaluates estimator. Logs
       train and test set evaluation metrics.
    Args:
      estimator (tf.Estimator):
      steps (int):
    Returns:
      None
    """
    estimator.train(input_fn(TRAIN_DATA), steps=steps)
    for path, data_type in list(zip([TRAIN_DATA, TEST_DATA], ['train', 'test'])):
        results = estimator.evaluate(input_fn(path), steps=steps)
        log_metrics(results, data_type)


# TODO:Figure out how to properly setup input_fn() and serving_input_fn()
# Other options:
# With lower-level APIs: https://stackoverflow.com/questions/33711556/making-predictions-with-a-tensorflow-model
# https://www.tensorflow.org/guide/saved_model#using_savedmodel_with_estimators
# def train_and_evaluate(estimator):
#     """Trains, evaluates, and exports the model"""
#     tf.summary.FileWriterCache.clear() # ensure filewriter cache is clear for TensorBoard events file
#     train_spec = tf.estimator.TrainSpec(
#         input_fn=input_fn(TRAIN_DATA),
#         max_steps=1000
#     )
#     exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
#     eval_spec = tf.estimator.EvalSpec(
#         input_fn=input_fn(TEST_DATA),
#         exporters=exporter)
#     # NOTE: Typically used for distributed training
#     return tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    tf.reset_default_graph()
    tf.logging.set_verbosity('ERROR')
    # NOTE: BoostedTreesClassifier does not support categorical columns with hash buckets
    shutil.rmtree(MODEL_DIR)  # Overwrites existing models

    create_train_test_datasets(test_size=0.2)
    classifier = build_estimator(MODEL_DIR)
    train_and_evaluate(classifier)
