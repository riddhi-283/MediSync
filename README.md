# MediSync
MediSync is a Streamlit application that leverages AI, powered by the LLMs, to analyze lab reports and assist in disease diagnosis. The application utilizes Gemini's OCR capabilities to accurately extract data from lab reports, enabling detailed analysis for diagnosis. It also incorporates OpenAI's GPT-3.5 Turbo to build a Retrieval-Augmented Generation (RAG) system, trained on a comprehensive dataset of diseases, symptoms, and treatments. This project is still in progress and till now, two functionalities have been added:

1) Lab Report Analysis
2) Disease Diagnosis on the basis of symptoms
   
Upcoming functionalities include disease diagnosis through body parts images, consultation with a virtual medical assistant, ability to answer medical examination questions with profound accuracy, etc.

# Results
![Screenshot 2024-10-19 015609](https://github.com/user-attachments/assets/a1892b11-253d-448f-b103-1daed2f62021)

![Screenshot 2024-10-19 021811](https://github.com/user-attachments/assets/b8123f6b-3803-4086-9b59-c5d3ac25989c)

![Screenshot 2024-10-19 021924](https://github.com/user-attachments/assets/8b30688b-f640-40bd-9078-dbef0a87901a)
![Screenshot 2024-10-19 024559](https://github.com/user-attachments/assets/e5caa42b-7134-4389-8c3b-a7fb5a06563f)
![Screenshot 2024-10-19 024613](https://github.com/user-attachments/assets/82272964-1cc9-4675-aad2-e314d52abf2a)
![Screenshot 2024-10-19 024625](https://github.com/user-attachments/assets/da586b69-4f01-49d9-a574-d263f1a68d88)




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
