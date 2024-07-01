from langchain.tools import tool, BaseTool
from src.tools.clean_text import *


class AcademicTool(BaseTool):
    name = "Academic Agent"
    description = '''
	Tool for fetching information from various academic articles. You will be given many papers to read through. Read these papers and images and report results.
	
	The format of this will be in a list of dictionaries, one per article. The format is as follows:
	-Name of the article
	-Text from the article
	-Images from the article
	    -This will be a dictionary, one with the title of the image and the other with the image matrix. Read these images and report results.

	This can answer questions such as:
    -What is the main hypothesis or research question addressed in {article name or nth acadmic article}
        -Find the right name through paper_name in the json. Look through the text of the article and summarize
    -Identify {n} key findings from {article or nth academic article}
        -Find the right name through paper_name in the json. Summarize the text and find {n} key findings
    -Summarize the methodology used in the {article or nth academic article}. Highlight any unique approaches or techniques played
        -Find the right name through paper_name in the json. Try to approach this from a scientific perspective, highlighting any groundbreaking ideas from the text
    -From the images and figures in {article} describe the trend in {figure}. What does it indicate about the research findings?
        -Find the right name through paper_name in the json. Look through the images given and identify which {figure} they are talking about based on paper_image_names in the json. Summarize what it describes.
    -Critically evalutate the statistical methods used in {article or nth article}. Are there any limitations or strengths worth noting?
        -Find the right name through paper_name in the json. Study the paper_text and deem its validity to the subject. You may have to refer to similar text references online for comparisons.
    -Integrate the findings from the {articles} to propose a new research direction or hypothesis. Justify your proposal based on the evidence provided in the {articles}.
        -Look at all of the paper_text outputs and highlight key findings, suggest options for further study, etc. 
    -Explain any images in detail
        -Find the paper_images associated with paper_image_names and summarize. Create and return your own visual if necessary.

	'''

    def __init__(self):
        super(AcademicTool, self).__init__()

    def _run(self):
        list_of_articles = ["attention", "Challenges LLM July 19_23","Continual_Pretraining",
                            "cs224n-2023-lecture11-prompting-rlhf","llm_review 2","Multimodal",
                            "Performance Evaluation","RAG Agent Resource-1"]
        response = get_article_info(list_of_articles, "../data/Academic_Papers")

        return response

    def _arun(self):
        print('not implemented, use _run')