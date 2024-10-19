import streamlit as st
from PIL import Image
import os
import google.generativeai as genai
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel("gemini-1.5-flash")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.environ['OPENAI_API_KEY'])
vectordb_file_path="faiss_index"
embeddings=OpenAIEmbeddings(model="text-embedding-3-large",api_key=os.environ['OPENAI_API_KEY'])

# Function to generate GPT's response for diagnosis
def format_symptoms(symptom_input):
    symptoms = [s.strip() for s in symptom_input.split(',')]
    formatted_symptoms = ', '.join(symptoms)
    return formatted_symptoms

def get_qa_chain(query):
    
    prompt_template = """
    You are a gentle, empathetic and expert medical assistant. A patient is experiencing the following symptoms: {question}.
    Below are several pieces of information retrieved from medical documents that match the patient's symptoms.

    Go through all the provided documents. Each document contains a disease and its related symptoms and treatments. 
    The disease name is present after "\nDisease:". Please follow these instructions:

    1. Group the symptoms related to each disease together.
    2. Write a diagnosis section for each disease, explaining how it affects the body.
    3. Write a treatment section where you explain all possible treatments for each disease, including medications, therapies, exercises, and dietary advice, if available.

    For example:
    - If you are experiencing **fever, nausea, and muscle pain**, you may have **Dengue Fever**. This is a viral infection transmitted by mosquitoes. It causes symptoms like high fever, joint pain, and fatigue. Treatments include hydration and pain relievers.
    - If you are experiencing **headache, confusion, and seizures**, you may have **Encephalitis**. This is an inflammation of the brain caused by viral infections. Treatments include antiviral medications and supportive care.
    
    Start every answer with an empathic tone like 'I am sorry to hear about that' or similiar sentences showing your concern towards patient's well-being. Also while answering about treatments, make it sound like "try doing..." or "you should...." or similiar tone to make sure that you sound concerned towards the user.

    You will be rewarded if you are able to list all diseases as per retrieved information along with their treatements and symptoms in thr above defined format.

    Do not add any additional knowledge beyond what is retrieved in the context.
    Only if no answer is found from extracted documents, then answer as per your expertise.

    CONTEXT: {context}

    QUESTION: {question}
    """

   
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    
    chain_type_kwargs = {"prompt": PROMPT}
    

    vectordb=FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()

    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

    # Get relevant documents using the retriever
    rdocs = retriever.get_relevant_documents(query)

    # Concatenate all retrieved document contents into a single context
    combined_context = "\n\n".join([doc.page_content for doc in rdocs])

    
    inputs = {
        "context": combined_context,  
        "question": query 
    }

    output = chain({"query": query, "context": combined_context})

    answer = output['result']  
    source_documents = output['source_documents']  

    print("Answer:")
    st.write("Answer:")
    print(answer)
    st.write(answer)


# Function to generate response using Gemini for lab reports
def get_response(input_text, image, prompt):
    response = model.generate_content([input_text, image[0], prompt])
    return response

def input_image_details(upload_file):
    if upload_file is not None:
        bytes_data = upload_file.getvalue()

        image_parts = [
            {
                "mime_type": upload_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")



# Initialize session state for image and input
if "selected_option" not in st.session_state:
    st.session_state.selected_option = None
    
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "input_text" not in st.session_state:
    st.session_state.input_text = ""



st.header("Medical Assistance App")
option = st.selectbox(
    "Select an option:",
    ["Select", "Lab Report Analysis", "Disease Diagnosis", "Talk with a Doctor"],
    index=0
)

if st.button("Confirm"):
    st.session_state.selected_option = option

if st.session_state.selected_option == "Lab Report Analysis":
        st.write("## Lab Report Analysis")
        upload_file = st.file_uploader("Please upload your lab report..", type=["jpg", "jpeg", "png"])

        if upload_file is not None:
            st.session_state.uploaded_image = Image.open(upload_file)
            st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_column_width=True)

        st.session_state.input_text = st.text_input("Input: ", key="input")

        submit = st.button("Tell me about the report")

        input_prompt = """
        You are a medical assistant who is very good at analyzing lab reports. Your job is to look at the lab report and find out which results are normal and which are not. For any result that is not normal, explain what it could mean for the person's health. Then, give easy-to-follow advice on how to improve those results, like suggesting foods to eat, exercises to do, or changes in daily habits.
        If the user asks about something specific like 'How is my heamoglobin' or 'Are my sodium levels normal' etc, then only answer that.
        You will be rewarded if you read the report correctly and your answer satisfies the user.
        """

        if submit and st.session_state.uploaded_image is not None:
            image_data = input_image_details(upload_file)
            response = get_response(st.session_state.input_text, image_data, input_prompt)
            st.subheader("Response:")
            st.write(response.text)
        elif submit and st.session_state.uploaded_image is None:
            st.warning("Please upload a lab report image.")


# Handle the Disease Diagnosis option
elif st.session_state.selected_option == "Disease Diagnosis":
    st.header("DIAGNOSIS")
    user_input = st.text_input("Enter symptoms for diagnosis (seperated by commas)")
    if user_input:
        formatted_output = format_symptoms(user_input)
        get_qa_chain(formatted_output)
    

# Handle the Talk with a Doctor option
elif st.session_state.selected_option == "Talk with a Doctor":
    st.write("## Talk with a Doctor")
    # st.write("Hello, I am your personalized medical assistant, how can I help you?")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ["Hello, I am your personalized medical assistant, how can I help you?"]

    # Display chat history
    for message in st.session_state.chat_history:
        st.write(f"Medical Assistant: {message}")

    user_input = st.text_input("Your message", "")

    if st.button("Send"):
        if user_input:
            st.session_state.chat_history.append(user_input)
            st.write(f"You: {user_input}")
            st.session_state.chat_history.append(f"Medical Assistant: Here's a response to your query: {user_input}")

# If no option is selected or none of the above options match
else:
    st.warning("Please select a valid option and confirm.")