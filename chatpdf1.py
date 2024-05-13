# import streamlit as st
# import requests
# from bs4 import BeautifulSoup
# from docx import Document
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))




# def get_text_from_docx(docx_files):
#     text = ""
#     for docx in docx_files:
#         doc = Document(docx)
#         for para in doc.paragraphs:
#             text += para.text + "\n"
#     return text

# def get_text_from_txt(txt_files):
#     text = ""
#     for txt in txt_files:
#         with open(txt, "r") as file:
#             text += file.read() + "\n"
#     return text

# def get_text_from_links(links):
#     text = ""
#     for link in links:
#         response = requests.get(link)
#         soup = BeautifulSoup(response.content, "html.parser")
#         paragraphs = soup.find_all("p")
#         for para in paragraphs:
#             text += para.text + "\n"
#     return text

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_from_sources(user_files):
#     text = ""
#     for file in user_files:
#         filename, extension = os.path.splitext(file.name)
#         if extension.lower() == ".pdf":
#             text += get_pdf_text([file])
#         elif extension.lower() == ".docx":
#             text += get_text_from_docx([file])
#         elif extension.lower() == ".txt":
#             text += get_text_from_txt([file])
#         elif extension.lower() in [".html", ".htm"]:
#             text += get_text_from_links([file])
#     return text

# # Inside your main() function





# def get_pdf_text(pdf_docs):
#     text=""
#     for pdf in pdf_docs:
#         pdf_reader= PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text+= page.extract_text()
#     return  text



# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")


# def get_conversational_chain():

#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-pro",
#                              temperature=0.3)

#     prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain



# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
#     new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization = True)
#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain()

    
#     response = chain(
#         {"input_documents":docs, "question": user_question}
#         , return_only_outputs=True)

#     print(response)
#     st.write("Reply: ", response["output_text"])




# def main():
#     st.set_page_config("Chat PDF")
#     st.header("Chat with PDF using Gemini💁")

#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_text_from_sources(pdf_docs)
#                 # raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")



# if __name__ == "__main__":
#     main()

import streamlit as st
import requests
from bs4 import BeautifulSoup
from docx import Document
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
# from google_auth_oauthlib import flow


# Define Google OAuth2 client information
# CLIENT_ID = 'your-client-id'
# CLIENT_SECRET = 'your-client-secret'
# SCOPES = ['openid', 'email', 'profile']

# def authenticate_with_google():
#     flow_args = {
#         'client_id': CLIENT_ID,
#         'client_secret': CLIENT_SECRET,
#         'scope': SCOPES,
#         'redirect_uri': 'postmessage'
#     }

#     flow_inst = flow.InstalledAppFlow.from_client_config(flow_args, scopes=SCOPES)
#     credentials = flow_inst.run_local_server()

#     return credentials


# st.set_page_config("Chat PDF")
# st.write("### Google API Key")
# st.write("You need a Google API Key to use this app.")
# st.write("If you don't have an API Key, you can create one [here](https://aistudio.google.com/app/apikey).")

# Prompt the user to enter the Google API Key
# api_key = st.text_input("Enter your Google API Key:", type='password')
# if api_key:
#     # Configure the Google API with the provided API Key
#     genai.configure(api_key=api_key)
#     # Remove the entire section regarding the Google API Key from the UI
#     st.empty()
#     st.empty()
#     st.empty()
# else:
#     st.warning("Please enter your Google API Key.")

# # Configure the Google API with the provided API Key
# # os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=api_key)


def get_text_from_docx(docx_files):
    text = ""
    for docx in docx_files:
        doc = Document(docx)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

def get_text_from_txt(txt_files):
    text = ""
    for txt in txt_files:
        with open(txt, "r") as file:
            text += file.read() + "\n"
    return text

def get_text_from_links(links):
    text = ""
    for link in links:
        response = requests.get(link)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        for para in paragraphs:
            text += para.text + "\n"
    return text

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_from_sources(user_files):
    text = ""
    for file in user_files:
        filename, extension = os.path.splitext(file.name)
        if extension.lower() == ".pdf":
            text += get_pdf_text([file])
        elif extension.lower() == ".docx":
            text += get_text_from_docx([file])
        elif extension.lower() == ".txt":
            text += get_text_from_txt([file])
        elif extension.lower() in [".html", ".htm"]:
            text += get_text_from_links([file])
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=st.session_state.api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3,google_api_key = st.session_state.api_key)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key = api_key)
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(api_key)

    response = chain({"input_documents":docs, "question": user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain(api_key)

#     response = chain({"input_documents":docs, "question": user_question}, return_only_outputs=True)

#     print(response)
#     st.write("Reply: ", response["output_text"])

# import streamlit as st
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai

def authenticate_api_key():
    st.title("Enter Google API Key")
    st.write("You need a Google API Key to use this app.")
    st.write("If you don't have an API Key, you can create one [here](https://aistudio.google.com/app/apikey).")

    # Prompt the user to enter the Google API Key
    api_key = st.text_input("Enter your Google API Key:", type='password')
    if api_key:
        # Configure the Google API with the provided API Key
        genai.configure(api_key=api_key)
        # Set API key in session state
        st.session_state.api_key = api_key
        # Redirect to "Upload Document" page
        st.experimental_rerun()
    return api_key

def upload_document():
    st.title("Upload Document")
    st.write("Upload your PDF files and click on the Submit button to process.")

    pdf_files = st.file_uploader("Upload PDF Files", accept_multiple_files=True)

    if pdf_files:
        st.write("PDFs Uploaded Successfully!")
        st.write(pdf_files)


def main():
    if "api_key" not in st.session_state:
        authenticate_api_key()
    else:
        st.title("Chat with PDF using Gemini🤖")
        st.markdown("<p style='text-align: center; font-size: 14px;'>done by 𝓥𝓪𝓻𝓾𝓷</p>", unsafe_allow_html=True)

        # Input field for user question
        user_question = st.text_input("Ask a Question from the PDF Files")

        # User question handling
        if user_question:
            user_input(user_question, st.session_state.api_key)

        # Sidebar menu
        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_text_from_sources(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")

if __name__ == "__main__":
    main()



# def main():
#     if not st.session_state.get("logged_in"):
#         st.title("Sign in with Google")
#         user = st.google_auth()
#         if user:
#             st.session_state.logged_in = True
#             st.success(f"Signed in as {user['email']}.")
#             # Proceed to the next page after authentication
#             st.experimental_rerun()
#     else:
#         if "api_key" not in st.session_state:
#             authenticate_api_key()
#         else:
#             st.title("Chat with PDF using Gemini🤖")
#             st.markdown("<p style='text-align: center; font-size: 14px;'>done by 𝓥𝓪𝓻𝓾𝓷</p>", unsafe_allow_html=True)

#             # Input field for user question
#             user_question = st.text_input("Ask a Question from the PDF Files")

#             # User question handling
#             if user_question:
#                 user_input(user_question, st.session_state.api_key)

#             # Sidebar menu
#             with st.sidebar:
#                 st.title("Menu:")
#                 pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#                 if st.button("Submit & Process"):
#                     with st.spinner("Processing..."):
#                         raw_text = get_text_from_sources(pdf_docs)
#                         text_chunks = get_text_chunks(raw_text)
#                         get_vector_store(text_chunks)
#                         st.success("Done")
#         if "api_key" in st.session_state:
#             upload_document()

# if __name__ == "__main__":
#     main()



# def main():
#     if "api_key" not in st.session_state:
#         authenticate_api_key()
#     else:
#         upload_document()
#         st.title("Chat with PDF using Gemini🤖")
#         st.markdown("<p style='text-align: center; font-size: 14px;'>done by 𝓥𝓪𝓻𝓾𝓷</p>", unsafe_allow_html=True)

#         # Input field for user question
#         user_question = st.text_input("Ask a Question from the PDF Files")

#         # User question handling
#         if user_question:
#             user_input(user_question)

#         # Sidebar menu
#         with st.sidebar:
#             st.title("Menu:")
#             pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#             if st.button("Submit & Process"):
#                 with st.spinner("Processing..."):
#                     raw_text = get_text_from_sources(pdf_docs)
#                     text_chunks = get_text_chunks(raw_text)
#                     get_vector_store(text_chunks)
#                     st.success("Done")


# if __name__ == "__main__":
#     main()


# Define your helper functions here
# def main():
#     # Check if API key is authenticated
#     api_key_authenticated = False

#     if "api_key" in st.session_state:
#         api_key_authenticated = True

#     if not api_key_authenticated:
#         st.title("Enter Google API Key")
#         st.write("You need a Google API Key to use this app.")
#         st.write("If you don't have an API Key, you can create one [here](https://aistudio.google.com/app/apikey).")

#         # Prompt the user to enter the Google API Key
#         api_key = st.text_input("Enter your Google API Key:", type='password')
#         if api_key:
#             # Configure the Google API with the provided API Key
#             genai.configure(api_key=api_key)
#             # Set API key in session state
#             st.session_state.api_key = api_key
#             # Redirect to "Chat with PDF" page
#             st.experimental_set_query_params(page="chat_with_pdf")
#             return

#     # Display "Chat with PDF" page
#     st.title("Chat with PDF using Gemini🤖")
#     st.markdown("<p style='text-align: center; font-size: 14px;'>done by 𝓥𝓪𝓻𝓾𝓷</p>", unsafe_allow_html=True)

#     # Input field for user question
#     user_question = st.text_input("Ask a Question from the PDF Files")

#     # User question handling
#     if user_question:
#         user_input(user_question)

#     # Sidebar menu
#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_text_from_sources(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")

# if __name__ == "__main__":
#     main()

# def main():
#     # Header
#     st.markdown("<h1 style='text-align: center;'>Chat with PDF using Gemini🤖</h1>", unsafe_allow_html=True)
#     st.markdown("<p style='text-align: center; font-size: 14px;'>done by 𝓥𝓪𝓻𝓾𝓷</p>", unsafe_allow_html=True)

#     # Input field for user question
#     user_question = st.text_input("Ask a Question from the PDF Files")

#     # User question handling
#     if user_question:
#         user_input(user_question)

#     # Sidebar menu
#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_text_from_sources(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")

# if __name__ == "__main__":
#     main()

