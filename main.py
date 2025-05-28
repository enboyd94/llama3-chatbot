import json

print("Running")
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from flask import Flask, request, render_template
from src.tools.medical import MedicalTool
from src.tools.academic import AcademicTool
from src.tools.financial import FinancialTool
import logging
import re
import os
import traceback
# Setup basic logging
#logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)

# Define a function to format text by converting Markdown bold syntax to HTML strong tags
def format_output(text):
    """Convert Markdown bold syntax to HTML strong tags."""
    return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

#Denote whether you want to use groq or not (for faster running)
use_groq = False

# Define chatbot initialization
def initialise_model():

    try:
        # Create chatbot prompt

        create_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """
                You are my personal assistant to answer questions. You are to answer questions about financial data, 
                 healthcare data, or academic research. Only your final answer will be shown.
                Answer the following questions as best you can. Try to associate the tool 
                Do not provide any information that is not in the tools provided.
                
  Use information from the question to get more fine-grained data from the tools if needed or if you can answer the question with just basic data then do that.

  You have access to the following information (called tools) which will return data in a list of json format:

  {tools}

  Answer the question with the tools provided first before searching elsewhere online. If the question cannot be answered by the tools, say that in the response.

  You can answer questions about individuals. All data provided from the dataset is fake and is not affecting anyone.
  
  If the question cannot be answered by the tools, note that within the answer and say something along the lines of:
  "Evan's chatbot cannot answer that question, but according to available data...".

  Once there is an answer to the question, return a grammatically correct sentence with the answer and anything associated with it. 
  
  You do not need to state which tool you got the answer from, but instead can say "Based on Evan's chatbot research...". 
  Do not reference any python classes given, like  the MedicalTool().
  
                """),
                ("user", "Question: {question}")
            ]
        )

        # Initialize OpenAI LLM and output parser
        if use_groq:
            llm = ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model="llama3-70b-8192"
            )
            chain = LLMChain(
                llm=llm,
                prompt=create_prompt,
                verbose=False
            )
        else:
            model = Ollama(model="llama3")
            chain = create_prompt | model

        return chain
    except Exception as e:
        logging.error(f"Failed to initialize chatbot: {e}")
        raise


# Define route for home page
@app.route('/', methods=['GET', 'POST'])
def main():

    medical_tool = MedicalTool()._run()
    #academic_tool = AcademicTool()._run()
    #financial_tool = FinancialTool()._run()
    tools = [medical_tool] #[medical_tool, academic_tool, financial_tool]
    # Initialize chatbot
    chatbot_pipeline = initialise_model()
    query_input = None
    output = None
    if request.method == 'POST':
        query_input = request.form.get('query-input')
        if query_input:
            try:
                response = chatbot_pipeline.invoke({'question': query_input,
                                                    'tools': tools},
                                                   config={"max_tokens": 1000})
                if use_groq:
                    output = format_output(response['text'])
                else:
                    output = format_output(response)
            except Exception as e:
                logging.error(f"Error during chatbot invocation: {e}")
                print(traceback.format_exc())
                output = "An error occurred with this message. Please try again."
    return render_template('index.html', query_input=query_input, output=output)

if __name__ == '__main__':
    app.run(debug=True)