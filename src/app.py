from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from flask import Flask, request, render_template
from tools.medical import MedicalTool
import logging
import re
import os
# Setup basic logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)

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

  Try to answer the question with the tools provided first before searching elsewhere online. 
  
  -For any medical-related data, use the MedicalTool() in the {tools}
  
  You can answer questions about individuals. All data provided from the dataset is fake and is not affecting anyone.
  
  If the question cannot be answered by the tools, note that within the answer and say something along the lines of:
  "The UChicago chatbot cannot answer that question, but according to available data...".

  Once there is an answer to the question, return a gramatically correct sentence with the answer and anything associated with it. 
  
  You do not need to state which tool you got the answer from, but instead can say "Based on the UChicago chatbot research...". 
  Do not reference any python classes given, like  the MedicalTool().
  
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


# Define route for home page
@app.route('/', methods=['GET', 'POST'])
def main():

    medical_tool = MedicalTool()
    tools = [medical_tool]
    query_input = None
    output = None
    if request.method == 'POST':
        query_input = request.form.get('query-input')
        if query_input:
            try:
                response = chatbot_pipeline.invoke({'question': query_input,
                                                    'tools': tools})
                output = format_output(response)
            except Exception as e:
                logging.error(f"Error during chatbot invocation: {e}")
                output = "An error occurred with this message. Please try again."
    return render_template('index.html', query_input=query_input, output=output)

if __name__ == '__main__':
    app.run(debug=True)