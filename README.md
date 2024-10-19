# MediSync
MediSync is a Streamlit application that leverages AI, powered by the LLMs, to analyze lab reports and assist in disease diagnosis. The application utilizes Gemini's OCR capabilities to accurately extract data from lab reports, enabling detailed analysis for diagnosis. It also incorporates OpenAI's GPT-3.5 Turbo to build a Retrieval-Augmented Generation (RAG) system, trained on a comprehensive dataset of diseases, symptoms, and treatments. This project is still in progress and till now, two functionalities have been added:

1) Lab Report Analysis
2) Disease Diagnosis on the basis of symptoms
   
Upcoming functionalities include disease diagnosis through body parts images, consultation with a virtual medical assistant, ability to answer medical examination questions with profound accuracy, etc.
 
# Installation
To run this project in your laptop:
```sh
git clone git@github.com:riddhi-283/MediSync.git
```
# Setting up the application
Get your free google-api-key from "makersuite.google.com" and an openai-api-key from "platform.openai.com"
<br> 
Create a .env file in the same folder and paste your both the api keys.
```sh
GOOGLE_API_KEY=''
OPENAI_API_KEY=''
```
Create a virtual environment either using pip or conda.

# Running the application
```sh
pip install -r requirements.txt
streamlit run combined.py
```
<br>
