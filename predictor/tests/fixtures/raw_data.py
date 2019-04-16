import pytest


@pytest.fixture(scope="session")
def raw_attendance():
    return ['No', 'no', 'NO', 'No ', 'no ', 'Yes', 'yes', 'yes ']


@pytest.fixture(scope="session")
def raw_industry_type():
    return ['Pharmaceuticals', 'IT Services', 'BFSI', 'Electronics',
            'Telecom', 'IT Products and Services', 'IT']


@pytest.fixture(scope="session")
def raw_interview_type():
    return ['Scheduled Walkin',
            'Scheduled ',
            'Walkin',
            'Scheduled Walk In',
            'Walkin ',
            'Sceduled walkin']


@pytest.fixture(scope="session")
def raw_client_name():
    return ['Hospira', 'UST', 'Standard Chartered Bank',
            'ANZ', 'Pfizer', 'Standard Chartered Bank Chennai',
            'Astrazeneca', 'Flextronics', 'Prodapt', 'Hewitt',
            'Williams Lea', 'Aon Hewitt', 'Aon hewitt Gurgaon',
            'Barclays', 'Woori Bank']


@pytest.fixture(scope="session")
def raw_question_response():
    return ['Yes',
            'No',
            'No- will take it soon',
            'Not yet',
            'Na',
            'Not Yet',
            'yes',
            'na']


@pytest.fixture(scope="session")
def raw_location():
    return ['Chennai',
            'Bangalore',
            'Hyderabad',
            'chennai',
            'Gurgaon',
            '- Cochin- ',
            'CHENNAI',
            'chennai ',
            'Gurgaonr',
            'Delhi',
            'Noida']


@pytest.fixture(scope="session")
def raw_position_skillset():
    return ['Production- Sterile',
            'Dot Net',
            'AML',
            'Trade Finance',
            'Routine',
            'Niche',
            'Selenium testing']


@pytest.fixture(scope="session")
def raw_candidate_skillset():
    return ['Routine',
            'Oracle',
            'Accounting Operations',
            'Banking Operations',
            'Fresher',
            'AML/KYC/CDD',
            'CDD KYC',
            'RA Label',
            'RA Publishing',
            'LCM -Manager',
            'Licensing – RA',
            'Biosimilars',
            'Analytical R & D',
            'Analytical R&D',
            'Senior software engineer-Mednet',
            'Tech lead-Mednet',
            'Technical Lead',
            'Sr Automation Testing',
            'Senior Analyst',
            'production',
            'Regulatory',
            'Core Java',
            'Oracle Plsql',
            'Automation Testing Java',
            'Submission Management',
            'Publishing',
            'Global Labelling',
            'ALS Testing',
            'Java Developer',
            'Lending and Liabilities',
            'Lending & Liability',
            'Lending And Liabilities',
            'Banking operations',
            'JAVA/J2EE/Struts/Hibernate',
            'JAVA/SPRING/HIBERNATE/JSF',
            'Java JSF',
            'Java,J2ee, JSF',
            'Java ,J2ee',
            'Java J2ee',
            '10.00 AM',
            '9.00 Am',
            'Java, J2Ee',
            'Java,J2EE',
            'Java/J2ee/Core Java',
            'Java',
            'Java/J2ee',
            'JAVA, J2ee',
            'T-24 developer',
            'Java J2EE',
            'COTS Developer',
            'Dot Net',
            'testing',
            'ETL',
            'Java-SAS',
            'Java Tech Lead',
            'SCCM',
            'SCCM-(Network, sharepoint,ms exchange)',
            'sccm',
            'SCCM – Sharepoint',
            'SAS',
            'Java, Spring, Hibernate',
            'Java,spring,hibernate',
            'Java, XML, Struts, hibernate',
            'Java ',
            'Java,SQL',
            'Biosimiliars',
            'EMEA',
            'Tech Lead- Mednet',
            'TL',
            'Production',
            'Biosimillar',
            'L & L',
            'Lending&Liablities',
            '11.30 AM',
            '12.30 Pm',
            '9.30 AM',
            '11.30 Am',
            'JAVA,J2ee',
            'Product Control',
            'COTS',
            '#NAME?',
            'Manager',
            'JAVA,SQL',
            'Java, SQL',
            'Hadoop',
            'SCCm- Desktop support',
            'Sccm- networking',
            'Production Support - SCCM',
            'BaseSAS Program/ Reporting',
            'JAVA/J2EE',
            'generic drugs – RA',
            'SCCM – SQL',
            '']
