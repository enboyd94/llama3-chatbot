# llama3-chatbot
This repo was used for a 2024 Hackathon sponsored by UChicago. The goal was to extract text, tables, and images from academic, financial, and medical files and use those as templates for Q&A for a chatbot.

To run this repo, install the required packages in requirements.txt.

```python
pip install -r requirements.txt
```

In addition, you will need to download Ollama here - https://ollama.com/download

Once that is downloaded onto your computer, navigate the terminal to this repo run the following command:

```python
ollama run llama3
```

This will set up llama3 onto your local machine.

There is an option to run the platfrom on groq instead of Ollama. You may download a key here - https://console.groq.com/keys

Switch the boolean use_groq = True in main.py to run through groq instead of Ollama.

To run the file locally, navigate the terminal to run main.py:

```python
python3 main.py
```

From there a link to a local run of the UI should appear. You may answer any questions associated with the academic, financial, and medical data. Enjoy!
