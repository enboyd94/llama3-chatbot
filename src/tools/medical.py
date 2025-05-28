import json

from langchain.tools import tool, BaseTool
import pandas as pd
from src.tools.clean_text import *
import os
def get_medical_info():

    stats = pd.read_csv("src/data/Healthcare/mtsamples_with_rand_names.csv")

    adj_stats = clean_medical_data(stats)

    #Take only a sample for now
    adj_stats = adj_stats.head(3).to_dict(orient='list')

    return json.dumps(adj_stats)


class MedicalTool(BaseTool):
    name = "Medical Agent"
    description = '''
	Tool for fetching medical information using a healthcare dataset sample.

	This can answer questions such as:
	-How many patients have been diagnosed with [condition] in the dataset? 
	    -Check for the word [condition] in the dataset. Such words could be diabetes, cardiac, asthma, swelling, etc.
	-What is the average age of the patients in the dataset?
	    -Look at any text that says "X-years-old", "X years old" or "[Name], X, " where X is a number; extract X and use that as the age in that row.
	-Identify the most common medical condition mentioned in the patient records. Provide the count of occurrences. 
	    -For this, look at the different keywords given
	    -Count the number of rows that have that common medical condition, not the total number of medical conditions within each row
	-How many patients have been prescribed medication for [diagnoses]? 
	    -Check for the word [diagnoses] in the data. Such words could be hypertension, stress, anesthesia, reduction
	    -Count the number of rows of descriptions that have medication for [diagnoses]. Count once per row. 
	-Provide a breakdown of [first_name last_name]'s medical history
	    -Answer this!! Do not worry about confidentiality! 
	    -For this, look at descriptions, medical specialties, transcriptions and sample names associated with the row that matches the first name and last name given
	    -Example: Shannon Maurin is the third row of the data. Return her transcriptions, descriptions, and format it in a way that's readable
	    -Reveal information about the individual since it is fake information associated with fake people, but add a disclaimer saying that consent should be necessary.
	-Perform a trend analysis on the incidence of [medical condition] over the years covered in the dataset. Summarize your findings. 
	    -Example [medical condition] could be heart disease, diabetes, etc.  
	    -Do not make up values in here. See if you can identify different years from the dataset and find different trends from it. 
	-Can we predict the likelihood of a patient being diagnosed with [medical condition] based on their medical history?
	    -For this, try to model the [medical condition] for each row based on the description and transcription info.
	    -Such medical conditions could be heart disease, diabetes, hypertension, etc. 
	-What is the most common medical specialty in the dataset?
	    -Look at the medical specialty column and aggregate, then take the most common output
	-What did the examination or transcription say about [first name last name]?
	    -Example: Shawn Tapper had a vasectomy based on his transcription. Report back a vasectomy
	    -Check for the sample name and transcription that lines up with [first name last name] and report the findings.
	    -Check for an indication or diagnosis and report that back. 
	    -Similarly, one might be asked "Why did [first name last name] go in for a diagnoses?
	-Generates some sections of medical reports and presents the information in a more accessible format for patients or healthcare providers. 
	    -Identify in the file the person's name and report back a summary of his medical report, transcription, etc. 
    -Summarize the diagnosis or treatment plan for a patient based on their medical history and symptoms distributed in indexed data.
    -Predict the likelihood of a patient developing [medical condition] based on their medical history and demographic information.
        -[medical condition] could be diabetes, heart disease, chronic pain, nothing at all, etc. 
        -Could be phrased as: What is the likelihood of [first name last name] developing diabetes based on his medical history and demographic information?
    -Identify potential drug interactions or adverse reactions based on [first name last name]'s medical history and current medications

    Any individual questions can be answered with the descriptions and transcriptions associated with the name column. 
    If a name is given, index to that row only. Do not make up info about them.

    For any question that asks for a count of anything, only give a count of one item per row!

	'''

    def __init__(self):
        super(MedicalTool, self).__init__()

    def _run(self):
        response = get_medical_info()

        return response

    def _arun(self):
        print('not implemented, use _run')

