# Predicting Interview Attendance
Using a [Kaggle dataset](https://www.kaggle.com/vishnusraghavan/the-interview-attendance-problem/data), this project develops an application that performs the following steps in order to predict if a candidate will attend an interview.
1. Preprocesses the raw data by cleaning and standardizing data fields that can be used as features in a predictive model. Creates two CSV files of labeled and unlabeled data (`preprocess/clean_data.py`).
2. Splits the labeled dataset into a training and test set using an 80%/20% split. Trains and evaluates a classifier (i.e. SVM) with scikit-learn using cross-validation for hyper-parameter tuning. Predicts interview attendance for candidates in the unlabeled dataset and writes the prediction results to a CSV file (`model/model.py`) .

## Setup
### Clone the project repo
`git clone git@github.com:jessdaubner/attendance-predictor.git`

### Build & Run the App
Download and install [Docker](https://www.docker.com/get-started) and build the app locally:
```bash
cd attendance-predictor
docker build -t attendance-predictor .
```

Running the container will execute all steps necessary to build the predictive model and generate predictions on the unlabeled data:
```
docker run --rm -ti attendance-predictor
```
Exepcted output:
```
08:35 $ docker run --rm -ti attendance-predictor
INFO | APP_DIR set to predictor
INFO | Writing new cleaned CSV with columns: ['candidate_id', 'client_name', 'industry_type', 'position_skillset', 'candidate_skillset', 'interview_type', 'gender', 'candidate_current_location', 'candidate_job_location', 'interview_venue', 'candidate_native_location', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'marital_status', 'attendance']
INFO | Wrote 1140 records to predictor/data/labeled.csv.
INFO | Wrote 93 records to predictor/data/unlabeled.csv.
INFO | APP_DIR set to predictor
INFO | Filled NAs of column client_name with STANDARD CHARTERED BANK
INFO | Filled NAs of column industry_type with BFSI
INFO | Filled NAs of column position_skillset with ROUTINE
INFO | Filled NAs of column candidate_skillset with JAVA
INFO | Filled NAs of column interview_type with SCHEDULED WALKIN
INFO | Filled NAs of column gender with MALE
INFO | Filled NAs of column candidate_current_location with CHENNAI
INFO | Filled NAs of column candidate_job_location with CHENNAI
INFO | Filled NAs of column interview_venue with CHENNAI
INFO | Filled NAs of column candidate_native_location with CHENNAI
INFO | Filled NAs of column q1 with YES
INFO | Filled NAs of column q2 with YES
INFO | Filled NAs of column q3 with YES
INFO | Filled NAs of column q4 with YES
INFO | Filled NAs of column q5 with YES
INFO | Filled NAs of column q6 with YES
INFO | Filled NAs of column q7 with YES
INFO | Filled NAs of column marital_status with SINGLE
INFO | Splitting labeled data into train and test set with 80/20 split
INFO | Training set created with 912 rows and 130 features
INFO | Test set created with 228 rows and 130 features
INFO | Fiting pipeline on training data set.........
INFO | ---------------------------
INFO | Test Set Evaluation Metrics
INFO | ---------------------------
INFO | Accuracy: 0.719298
INFO | Baseline Accuracy: 0.640351
INFO | AUC: 0.617775
INFO | Average Precision: 0.699735
INFO | ---------------------------
INFO | Writing svm-model-04-16-2019.joblib to predictor/model/exported/
INFO | Making predictions on unlabeled data.........
INFO | Filled NAs of column client_name with STANDARD CHARTERED BANK
INFO | Filled NAs of column industry_type with BFSI
INFO | Filled NAs of column position_skillset with ROUTINE
INFO | Filled NAs of column candidate_skillset with JAVA
INFO | Filled NAs of column interview_type with SCHEDULED WALKIN
INFO | Filled NAs of column gender with MALE
INFO | Filled NAs of column candidate_current_location with CHENNAI
INFO | Filled NAs of column candidate_job_location with CHENNAI
INFO | Filled NAs of column interview_venue with CHENNAI
INFO | Filled NAs of column candidate_native_location with CHENNAI
INFO | Filled NAs of column q1 with YES
INFO | Filled NAs of column q2 with YES
INFO | Filled NAs of column q3 with YES
INFO | Filled NAs of column q4 with YES
INFO | Filled NAs of column q5 with YES
INFO | Filled NAs of column q6 with YES
INFO | Filled NAs of column q7 with YES
INFO | Filled NAs of column marital_status with SINGLE
INFO | Wrote predictions for 93 candidates to predictor/data/predictions.csv```

NOTE: This will overwrite some of the existing data files and model artifacts in the container.

### Running Jupyter Notebooks
To explore the raw or cleaned CSV files in the `predictor/data/` including `labeled.csv`, `unlabeled.csv`, and `predictions.csv`, launch a notebook from the container:
1. Run the container as follows:
```bash
docker run -it -p 8888:8888 attendance-predictor
```
2. At the command prompt inside the container:
```bash
cd predictor
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```
3. Follow the instructions beneath the prompt (_To access the notebook, open this file in a browser:..._) and copy the URL into a browser tab.

### Development
To make development faster, volume mount the `app` directory over the one in the container:
```bash
docker run --rm -v `pwd`/predictor:/predictor -ti attendance-predictor /bin/bash
```
This enables edits made locally to be reflected in the container and be quickly tested and run.

### Testing
To run unit tests, run `pytest` within the project.
```
root@4689da80e66d:/predictor# pytest
========================================================================= test session starts =========================================================================
platform linux -- Python 3.7.3, pytest-4.4.1, py-1.8.0, pluggy-0.9.0
rootdir: /predictor
collected 6 items

tests/preprocess/test_clean_data.py ......                                                                                                                      [100%]

====================================================================== 6 passed in 0.39 seconds =======================================================================
```

## Model

### Explaining Model Performance
Probability is a measure of how likely an event is to occur. However, the interpretation of specific probabilities can be subjective depending on the person interpreting the value. For example, a probability of 30% could be interpreted as falling into different areas of possibility, such as "probably not" or "highly doubtful" depending on the interpreter if the information is provided in a context that lacks agreed upon definitions mapping mathematical odds to commonly used phrases of probability. While a 30% chance of attendance may be casually understood as an indication that a candidate is highly unlikely to attend, another perspective is that over the long-run or given a larger sample size given a candidate and position with the same set of attributes, we'd expect the candidate to attend the interview approximately 30 times out of 100.

Given the inherent subjectivity in interpreting probabilities for disparate use cases, constructing a confusion matrix and an ROC curve for the model enables us to better understand and evaluate model performance than the probability of attendance generated by the model alone. The confusion matrix tabulates the false positives, false negatives, true positives, and true negatives generated by the model per possible outcome. For example, if users are particularly sensitive to expected "no-show" candidates arriving onsite, or false negatives, we'll want to minimize the false negative rate and adjust the threshold used to predict attendance to err towards minimizing false negatives. The ROC curve enables us to do adjust the threshold in this manner by plotting the true positive rate against the false positive rate at varying thresholds. AUC, or area under the curve of the ROC curve, is also commonly used to evaluate classifiers; scores range from 0.5 to 1.0, with higher scores indicating a better classifier.

## API
Managed services like AWS SageMaker or GCP's ML Engine can be used to create a scalable RESTful API for inference. Specifically, a trained scikit-learn pipeline can be saved as a pickle or joblib file and uploaded to S3 or Cloud Storage. In the case of AWS SageMaker, the exported model parameters can be injected into SageMaker's first-party containers for prediction. Then a SageMaker endpoint specifying the hosted model and resources to be deployed can be configured and created to serve the model and receive inference requests. In Python, these steps can be executed using `boto3`, the low-level AWS SDK, or `sagemaker`, a high-level library for deploying models on SageMaker.

## Resources
* [Words of Estimative Probability - CIA Library](https://www.cia.gov/library/center-for-the-study-of-intelligence/csi-publications/books-and-monographs/sherman-kent-and-the-board-of-national-estimates-collected-essays/6words.html)

### API
* [AWS SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk)
* [Deploy the Model to Amazon SageMaker Hosting Services](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-deploy-model.html)
* [GCP - Predictions with scikit-learn pipelines](https://cloud.google.com/ml-engine/docs/scikit/using-pipelines)

### Scikit Learn
* [Parameter estimation using grid search with cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html)
* [Model Persistence](https://scikit-learn.org/stable/modules/model_persistence.html)
