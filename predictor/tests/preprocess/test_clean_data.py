import pytest

from predictor.preprocess.clean_data import clean_string, format_industry_type, format_interview_type, \
    format_question_response, format_client_name, format_candidate_skillset


def test_clean_string():
    assert clean_string('NA') is None
    assert clean_string('na') is None
    assert clean_string('- Cochin- ') == 'COCHIN'
    assert clean_string('chennai') == 'CHENNAI'
    assert clean_string('yes ') == 'YES'
    assert clean_string('yes ') == 'YES'
    assert clean_string('Java,J2ee, JSF') == 'JAVA J2EE  JSF'


@pytest.mark.usefixtures('raw_industry_type')
@pytest.mark.usefixtures('clean_industry_type')
def test_format_industry_type(raw_industry_type, clean_industry_type):
    cleaned_values = [format_industry_type(clean_string(i)) for i in raw_industry_type]
    assert len(list(filter(lambda x: x == 'IT', cleaned_values))) == 3
    assert len(set(cleaned_values)) == len(clean_industry_type)


@pytest.mark.usefixtures('raw_interview_type')
@pytest.mark.usefixtures('clean_interview_type')
def test_format_interview_type(raw_interview_type, clean_interview_type):
    cleaned_values = [format_interview_type(clean_string(i)) for i in raw_interview_type]
    assert len(set(cleaned_values)) == len(clean_interview_type)


@pytest.mark.usefixtures('raw_client_name')
@pytest.mark.usefixtures('clean_client_name')
def test_format_client_name(raw_client_name, clean_client_name):
    cleaned_values = [format_client_name(clean_string(i)) for i in raw_client_name]
    assert len(list(filter(lambda x: x == 'AON HEWITT', cleaned_values))) == 3
    assert len(list(filter(lambda x: x == 'STANDARD CHARTERED BANK', cleaned_values))) == 2
    assert len(set(cleaned_values)) == len(clean_client_name)


@pytest.mark.usefixtures('raw_question_response')
@pytest.mark.usefixtures('clean_question_response')
def test_format_question_repsonse(raw_question_response, clean_question_response):
    cleaned_values = [format_question_response(clean_string(i)) for i in raw_question_response]
    non_null_count = len(set(list(filter(lambda x: x is not None, cleaned_values))))
    assert non_null_count == len(clean_question_response)


@pytest.mark.usefixtures('raw_candidate_skillset')
def test_format_candidate_skillset(raw_candidate_skillset):
    assert format_candidate_skillset('SCCM  DESKTOP SUPPORT') == 'SCCM'
    java_values = ['CORE JAVA', 'JAVA  J2EE']
    assert len(set([format_candidate_skillset(i) for i in java_values])) == 1
    assert format_candidate_skillset('SAS') == 'SAS'
    assert format_candidate_skillset('ETL') == 'ETL'
    assert format_candidate_skillset('ETL') == 'ETL'
