"""
Cleans data from original CSV file and produces CSV files containing
labeled and unlabeled data
"""
import re
import csv
from dataclasses import dataclass, astuple

from predictor.config import logger, RAW_DATA, CSV_COLUMNS, \
    LABELED_DATA, UNLABELED_DATA


def write_csv(data, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(data)

        n_records = len(data)
        logger.info(f'Wrote {n_records} records to {filename}.')


def clean_string(text):
    """Removes whitespace, upper-cases text, and formats 'NA'
       and '' values as None.
    """
    replaced_text = re.sub('[-/—,]', ' ', text)
    cleaned_text = replaced_text.upper().strip()

    if not cleaned_text or cleaned_text == 'NA':
        return None
    else:
        return cleaned_text


def format_interview_type(interview_type):
    """Correct typos in 'Observed Attendance' data to one of
       four acceptable values: None, SCHEDULED WALKIN, SCHEDULED,
       and WALKIN.
    """
    if interview_type in ['SCHEDULED WALKIN', 'SCHEDULED WALK IN', 'SCEDULED WALKIN']:
        return 'SCHEDULED WALKIN'
    else:
        return interview_type


def format_client_name(client_name):
    if not client_name:
        return None
    elif 'STANDARD CHARTERED' in client_name:
        return 'STANDARD CHARTERED BANK'
    elif 'HEWITT' in client_name:
        return 'AON HEWITT'
    else:
        return client_name


def format_industry_type(industry_type):
    if not industry_type:
        return None
    elif 'IT' in industry_type:
        return 'IT'
    else:
        return industry_type


def format_question_response(response):
    if response == 'YES':
        return 'YES'
    elif response and response != 'YES':
        return 'NO'
    else:
        None


def format_candidate_skillset(skillset):
    # TODO: Tokenize instead of hardcoding
    if not skillset:
        return None
    elif skillset == 'FRESHER':
        return 'ROUTINE'
    elif 'JAVA' in skillset:
        return 'JAVA'
    elif 'BIO' in skillset:
        return 'BIOSIMILAR'
    elif 'SCCM' in skillset:
        return 'SCCM'
    elif 'ANALYTICAL' in skillset:
        return 'ANALYTICAL R&D'
    elif 'COTS' in skillset:
        return 'COTS'
    elif 'MEDNET' in skillset:
        return 'MEDNET'
    elif 'RA' in skillset or 'PUBLISHING' in skillset or 'LABEL' in skillset:
        return 'REGULATORY'
    elif 'CDD' in skillset:
        return 'CDD'
    elif 'ORACLE' in skillset:
        return 'ORACLE'
    elif 'LENDING' in skillset or 'L &'in skillset:
        return 'LENDING'
    elif (' AM' in skillset) or (' PM' in skillset) or (skillset == '#NAME?'):
        return None
    elif ('MANAGE' in skillset) or ('LEAD' in skillset):
        return 'MANAGEMENT'
    elif 'TEST' in skillset:
        return 'TESTING'
    else:
        return skillset


@dataclass
class DataRecord:
    """Class for keeping track of and initiating cleaning of a data record"""
    candidate_id: str
    client_name: str
    industry_type: str
    position_skillset: str
    candidate_skillset: str
    interview_type: str
    gender: str
    candidate_current_location: str
    candidate_job_location: str
    interview_venue: str
    candidate_native_location: str
    q1: str
    q2: str
    q3: str
    q4: str
    q5: str
    q6: str
    q7: str
    marital_status: str
    attendance: str

    def __post_init__(self):
        self.client_name = format_client_name(self.client_name)
        self.industry_type = format_industry_type(self.industry_type)
        self.interview_type = format_interview_type(self.interview_type)
        self.candidate_skillset = format_candidate_skillset(self.candidate_skillset)
        self.q1 = format_question_response(self.q1)
        self.q2 = format_question_response(self.q2)
        self.q3 = format_question_response(self.q3)
        self.q4 = format_question_response(self.q4)
        self.q5 = format_question_response(self.q5)
        self.q6 = format_question_response(self.q6)
        self.q7 = format_question_response(self.q7)


def clean_data():
    with open(RAW_DATA) as file:
        reader = csv.reader(file, delimiter=',')
        # Skip header row
        next(reader, None)

        unlabeled_data = []
        labeled_data = []
        for row in reader:
            row = [clean_string(value) for value in row]
            # NOTE: Interview Date and Candidate location are dropped because it duplicates Candiate Current Location except for a typo
            record = DataRecord(client_name=row[1],
                                industry_type=row[2],
                                position_skillset=row[4],
                                candidate_skillset=row[5],
                                interview_type=row[6],
                                candidate_id=row[7],
                                gender=row[8],
                                candidate_current_location=row[9],
                                candidate_job_location=row[10],
                                interview_venue=row[11],
                                candidate_native_location=row[12],
                                q1=row[13],
                                q2=row[14],
                                q3=row[15],
                                q4=row[16],
                                q5=row[17],
                                q6=row[18],
                                q7=row[19],
                                marital_status=row[21],
                                attendance=row[20])

            if record.client_name:
                if record.attendance:
                    labeled_data.append(astuple(record))
                else:
                    unlabeled_data.append(astuple(record))
        return labeled_data, unlabeled_data


if __name__ == '__main__':
    labeled_data, unlabeled_data = clean_data()
    logger.info(f'Writing new cleaned CSV with columns: {CSV_COLUMNS}')
    write_csv(labeled_data, LABELED_DATA)
    write_csv(unlabeled_data, UNLABELED_DATA)
