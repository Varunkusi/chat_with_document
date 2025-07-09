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

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3,google_api_key = st.session_state.api_key)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# def user_input(user_question, api_key):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key = api_key)
    
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain(api_key)

#     response = chain({"input_documents":docs, "question": user_question}, return_only_outputs=True)

#     print(response)
#     st.write("Reply: ", response["output_text"])

def user_input(user_question,api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",google_api_key = api_key)  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(api_key)

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response

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
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

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
        st.rerun()
    return api_key

def upload_document():
    st.title("Upload Document")
    st.write("Upload your PDF files and click on the Submit button to process.")

    pdf_files = st.file_uploader("Upload PDF Files", accept_multiple_files=True)

    if pdf_files:
        st.write("PDFs Uploaded Successfully!")
        st.write(pdf_files)



def main():
    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon="🤖"
    )

    if "api_key" not in st.session_state or not st.session_state.api_key:
        authenticate_api_key()
        return

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_text_from_sources(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Main content area for displaying chat messages
    st.title("Chat with PDF files using Gemini🤖")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt, st.session_state.api_key)  # Pass api_key argument here
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
