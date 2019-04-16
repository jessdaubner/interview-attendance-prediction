import pytest
from predictor.preprocess.vocabulary import VOCABULARY


@pytest.fixture(scope="session")
def clean_attendance():
    return VOCABULARY['attendance']


@pytest.fixture(scope="session")
def clean_industry_type():
    return VOCABULARY['industry_type']


@pytest.fixture(scope="session")
def clean_interview_type():
    return VOCABULARY['interview_type']


@pytest.fixture(scope="session")
def clean_interview_venue():
    return VOCABULARY['interview_venue']


@pytest.fixture(scope="session")
def clean_client_name():
    return VOCABULARY['client_name']


@pytest.fixture(scope="session")
def clean_question_response():
    return VOCABULARY['q1']


@pytest.fixture(scope="session")
def clean_candidate_current_location():
    return VOCABULARY['candidate_current_location']


@pytest.fixture(scope="session")
def clean_candidate_job_location():
    return VOCABULARY['candidate_job_location']


@pytest.fixture(scope="session")
def clean_candidate_native_location():
    return VOCABULARY['candidate_native_location']


@pytest.fixture(scope="session")
def clean_position_skillset():
    return VOCABULARY['position_skillset']


@pytest.fixture(scope="session")
def clean_candidate_skillset():
    return VOCABULARY['candidate_skillset']
