from langchain.tools import tool, BaseTool
from src.tools.clean_text import *



class FinancialTool(BaseTool):
    name = "Academic Agent"
    description = '''
	Tool for fetching information from various academic articles. You will be given many papers to read through. Read these papers and images and report results.

	The format of this will be in a list of dictionaries, one per article. The format is as follows:
	-Name of the article
	-Text from the article
	-Image names from the article
	-Images details from the article
	    -This will be a dictionary, one with the title of the image and the other with the image matrix. Read these images and report results.

	This can answer questions such as:
	-What was the total revenue reported by {company} in {year}?
	    -{company} could be Google (alphabet), Facebook, NVIDIA, or Apple
	    -{year} can be a year since 2020, last fiscal year, current fiscal year, etc. The article title should provide the right year, so will the text details
	    -Look for dollar amounts throughout the text associated with revenue
	-Name {n} key products or services that significantly contributed to {company}'s revenue growth.
	    -{n} is a number, {company} could be Google (alphabet), Facebook, NVIDIA, or Apple
	    -Look at the texts and summarize {n} key findings from the text based on that {company}
	-Compare the year-over-year revenue growth of {companies}. Which company showed a higher growth rate?
	    -{companies} could be Google (alphabet), Facebook, NVIDIA, or Apple. More than one company could be named here
	    -Look through all texts and compare different revenue growths per {companies}
	-Identify the primary cost drivers for {company} as detailed in the financial documents
	-Analyze the financial health of {companies} using key financial ratios like P/E ratio, debt-to-equity ratio and return on equity
	-What strategic initiatives has {company} undertaken to enhance its market position?
	    -Provide specific examples here within the texts associated with {company}
	-Accurately extract and summarize key financial metrics from the reports, such as revenue, profit margins and earnings per share.
	-Answer trends and comparisons across different quarters or fiscal years
	-Generate a comprehensive financial summary by synthesizing the data from the tables and figures
	    -Look through the numbers on both the text and images and report accurate info
	-Identify and flag potential discrepancies or anomalies in financial statements, such as unusual expense patterns or revenue inconsistencies
	-Infer and compare financial performances based on the data and provide insights into growth or risks for {company}


	'''

    def __init__(self):
        super(FinancialTool, self).__init__()

    def _run(self):
        list_of_articles = ["2020_alphabet_annual_report", "2021_alphabet_annual_report", "2022-alphabet-annual-report",
                            "2023_alphabet", "2024_alphabet-10-q-q1-2024", "Amazon-2020-Annual-Report",
                            "Amazon-2021-Annual-Report", "Amazon-2022-Annual-Report",
                            "Amazon-com-Inc-2023-Annual-Report", "Apple_10-K-2021", "Apple_10-K-Q4-2020",
                            "Apple_10-K-Q4-2022", "Apple_10-K-Q4-2023", "FB_2020-Annual-Report_FB",
                            "FB_2021-Annual-Report_FB", "FB_2022", "FB_2023", "NASDAQ_NVDA_2020", "NASDAQ_NVDA_2021",
                            "NASDAQ_NVDA_2022", "NASDAQ_NVDA_2023"]
        response = get_article_info(list_of_articles, "../data/Financial_Documents")

        return response

    def _arun(self):
        print('not implemented, use _run')