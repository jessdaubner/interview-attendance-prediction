"""
Defines the vocabulary (i.e, the expected, cleaned values, excluding
null values) after initial pre-processing and standardization for each feature.
"""
# TODO: Generate these automatically with df[column_name].unique()
INTERVIEW_TYPE = ['SCHEDULED', 'WALKIN', 'SCHEDULED WALKIN']
INDUSTRY_TYPE = ['IT', 'BFSI', 'PHARMACEUTICALS', 'TELECOM', 'ELECTRONICS']
GENDER = ['MALE', 'FEMALE']
MARITAL_STATUS = ['SINGLE', 'MARRIED']
YES_NO = ['YES', 'NO']
CLIENT_NAME = ['HOSPIRA',
               'UST',
               'STANDARD CHARTERED',
               'ANZ',
               'PFIZER',
               'ASTRAZENECA',
               'FLEXTRONICS',
               'PRODAPT',
               'AON HEWITT',
               'WILLIAMS LEA',
               'BARCLAYS',
               'WOORI BANK']

POSITION_SKILLSET = ['PRODUCTION  STERILE', 'DOT NET', 'AML', 'TRADE FINANCE',
                     'ROUTINE', 'NICHE', 'SELENIUM TESTING']

CANDIDATE_SKILLSET = [
    'ROUTINE',
    'REGULATORY',
    'CDD',
    'MANAGEMENT',
    'BIOSIMILAR',
    'ANALYTICAL R&D',
    'MEDNET',
    'TESTING',
    'SENIOR ANALYST',
    'PRODUCTION',
    'JAVA',
    'LENDING',
    'T 24 DEVELOPER',
    'COTS',
    'DOT NET',
    'SCCM',
    'SAS',
    'EMEA',
    'PRODUCT CONTROL',
    'HADOOP']
CANDIDATE_NATIVE_LOCATION = ['HOSUR',
                             'TRICHY',
                             'CHENNAI',
                             'TRIVANDRUM',
                             'BANGALORE',
                             'COCHIN',
                             'COIMBATORE',
                             'MUMBAI',
                             'HYDERABAD',
                             'PUNE',
                             'ALLAHABAD',
                             'KOLKATA',
                             'CUTTACK',
                             'GURGAON',
                             'VISAKAPATINAM',
                             'PATNA',
                             'CHITOOR',
                             'WARANGAL',
                             'VIJAYAWADA',
                             'SALEM',
                             'VELLORE',
                             'NAGERCOIL',
                             'BHUBANESHWAR',
                             'BADDI',
                             'CHANDIGARH',
                             'TIRUPATI',
                             'KANPUR',
                             'AMBUR',
                             'NOIDA',
                             'AGRA',
                             'DELHI  NCR',
                             'TANJORE',
                             'ANANTAPUR',
                             'AHMEDABAD',
                             'KURNOOL',
                             'PONDICHERRY',
                             'GHAZIABAD',
                             'FAIZABAD',
                             'MYSORE',
                             'HISSAR',
                             'DELHI',
                             'LUCKNOW']

INTERVIEW_VENUE = ['HOSUR', 'BANGALORE', 'CHENNAI', 'HYDERABAD', 'GURGAON', 'COCHIN', 'NOIDA']
CANDIDATE_CURRENT_LOCATION = ['CHENNAI', 'BANGALORE', 'HYDERABAD', 'GURGAON', 'COCHIN', 'DELHI', 'NOIDA']
CANDIDATE_JOB_LOCATION = ['HOSUR', 'BANGALORE', 'CHENNAI', 'VISAKAPATINAM', 'GURGAON', 'COCHIN', 'NOIDA']

VOCABULARY = {
    'client_name': CLIENT_NAME,
    'industry_type': INDUSTRY_TYPE,
    'position_skillset': POSITION_SKILLSET,
    'candidate_skillset': CANDIDATE_SKILLSET,
    'interview_type': INTERVIEW_TYPE,
    'gender': GENDER,
    'candidate_current_location': CANDIDATE_CURRENT_LOCATION,
    'candidate_job_location': CANDIDATE_JOB_LOCATION,
    'interview_venue': INTERVIEW_VENUE,
    'candidate_native_location': CANDIDATE_NATIVE_LOCATION,
    'q1': YES_NO,
    'q2': YES_NO,
    'q3': YES_NO,
    'q4': YES_NO,
    'q5': YES_NO,
    'q6': YES_NO,
    'q7': YES_NO,
    'marital_status': MARITAL_STATUS,
}

VOCABULARY_SORTED = {k: sorted(v) for k, v in VOCABULARY.items()}
