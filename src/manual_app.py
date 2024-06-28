##This is if you do not want to use the flask app
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools.medical import MedicalTool
from sentence_transformers import SentenceTransformer #(package is sentence-transformers)
import logging
import re
# Setup basic logging
logging.basicConfig(level=logging.DEBUG)


# Define a function to format text by converting Markdown bold syntax to HTML strong tags
def format_output(text):
    """Convert Markdown bold syntax to HTML strong tags."""
    return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)


# Define chatbot initialization
def initialise_llama3():

    try:
        # Create chatbot prompt

        create_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """
                You are my personal assistant to answer questions. You are to anwser questions about financial data, 
                 healthcare data, or academic research. Only your final answer will be shown.
                Answer the following questions as best you can. Try to associate the tool 
                Do not provide any information that is not in the tools provided.

  Use information from the question to get more fine-grained data from the tools if needed or if you can answer the question with just basic data then do that.

  You have access to the following tools which will return data in a json format:

  {tools}

  If you use the Opta Widget Supercomputer tool, just return its json output IMMEDIATELY. DO NOT change or add anything in its output.Dont tell
  "This is the output". Just return what the widget tool returned.

  Use the following format:

  Question: the input question you must answer
  Thought: you should always think about what to do
  Action: the action to take, should be one of [{tools}]
  Action Input: the input to the action
  Observation: the result of the action
	(this Thought/Action/Action Input/Observation can repeat 3 times)

  When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format and be a valid JSON:

                """),
                ("user", "Question: {question}")
            ]
        )

        # Initialize OpenAI LLM and output parser
        llama_model = Ollama(model="llama3")
        format_output = StrOutputParser()

        # Create chain
        chatbot_pipeline = create_prompt | llama_model | format_output
        return chatbot_pipeline
    except Exception as e:
        logging.error(f"Failed to initialize chatbot: {e}")
        raise


# Initialize chatbot
chatbot_pipeline = initialise_llama3()

def transform_sentence(tools_to_use):

    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose a model that fits your requirements
    text_embedding = sentence_model.encode(tools_to_use)

    return text_embedding


def main():
    medical_tool = MedicalTool()
    tools = [medical_tool]

    embedded_tools = transform_sentence(tools)

    query_input = "How many patients have been diagnosed with diabetes in the dataset?"
    try:
        response = chatbot_pipeline.invoke({'question': query_input,
                                            'tools': embedded_tools})
        output = format_output(response)
    except Exception as e:
        logging.error(f"Error during chatbot invocation: {e}")
        output = "An error occurred with this message. Please try again."

    print(output)
    return output
if __name__ == '__main__':
    main()