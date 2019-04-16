"""Global variables used across programs"""
import os
import sys
import logging

# Configure logging
logger = logging.getLogger('predictor')
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(fmt='%(levelname)s | %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.setLevel('INFO')

# Data file locations
APP_DIR = os.getenv('APP_DIR')
logger.info(f'APP_DIR set to {APP_DIR}')

DATA_DIR = f'{APP_DIR}/data'
RAW_DATA = f'{DATA_DIR}/Interview_Attendance_Data.csv'
LABELED_DATA = f'{DATA_DIR}/labeled.csv'
UNLABELED_DATA = f'{DATA_DIR}/unlabeled.csv'
PREDICTIONS = f'{DATA_DIR}/predictions.csv'
TEST_DATA = f'{DATA_DIR}/train.csv'
TRAIN_DATA = f'{DATA_DIR}/test.csv'
MODEL_DIR = f'{APP_DIR}/model/exported/'

# CSV file descriptors and column header of pre-proceessed labeled and unlabeled datasets
CSV_COLUMNS = [
    'candidate_id',
    'client_name',
    'industry_type',
    'position_skillset',
    'candidate_skillset',
    'interview_type',
    'gender',
    'candidate_current_location',
    'candidate_job_location',
    'interview_venue',
    'candidate_native_location',
    'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7',
    'marital_status',
    'attendance',
]

# Map of raw CSV file column names to cleaned CSV files
COLUMN_HEADER_MAPPING = {
    'Date of Interview': 'interview_date',
    'Client name': 'client_name',
    'Industry': 'industry_type',
    'Location': 'candidate_location',
    'Position to be closed': 'position_skillset',
    'Nature of Skillset': 'candidate_skillset',
    'Interview Type': 'interview_type',
    'Name(Cand ID)': 'candidate_id',
    'Gender': 'gender',
    'Candidate Current Location': 'candidate_current_location',
    'Candidate Job Location': 'candidate_job_location',
    'Interview Venue': 'interview_venue',
    'Candidate Native location': 'candidate_native_location',
    'Have you obtained the necessary permission to start at the required time': 'q1',
    'Hope there will be no unscheduled meetings': 'q2',
    'Can I Call you three hours before the interview and follow up on your attendance for the interview': 'q3',
    'Can I have an alternative number/ desk number. I assure you that I will not trouble you too much': 'q4',
    'Have you taken a printout of your updated resume. Have you read the JD and understood the same': 'q5',
    'Are you clear with the venue details and the landmark.': 'q6',
    'Has the call letter been shared': 'q7',
    'Observed Attendance': 'attendance',
    'Marital Status': 'marital_status'
}

# Specify column used in  model training
LABEL = 'attendance'

FEATURES = [column for column in CSV_COLUMNS
            if column not in (LABEL, 'candidate_id')]
