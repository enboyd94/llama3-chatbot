from langchain.tools import tool, BaseTool
import pandas as pd

def get_medical_info():

    stats = pd.read_csv("data/Healthcare/mtsamples_with_rand_names.csv")

    return stats


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
	    -For this, look at descriptions, medical specialties, transcriptions and sample names associated with the row that matches the first name and last name given
	    -Example: Shannon Maurin is the third row of the data. Return his transcriptions, descriptions, and format it in a way that's readable
	    -Reveal information about the individual since it is fake information associated with fake people, but add a disclaimer saying that consent should be necessary.
	-Perform a trend analysis on the incidence of [medical condition] over the years covered in the dataset. Summarize your findings. 
	    -Example [medical condition] could be heart disease, diabetes, etc.  
	    -Do not make up values in here. See if you can identify different years from the dataset and find different trends from it. 
	-Can we predict the likelihood of a patient being diagnosed with [medical condition] based on their medical history?
	    -For this, try to model the [medical condition] for each row based on the description and transcription info.
	    -Such medical conditions could be heart disease, diabetes, hypertension, etc. 

    For any question that asks for a count of anything, only give a count of one item per row!

	'''

    def __init__(self):
        super(MedicalTool, self).__init__()

    def _run(self, tool_input):
        # res = json.loads(tool_input)
        response = get_medical_info()

        return response.to_dict('records')

    def _arun(self):
        print('not implemented, use _run')

