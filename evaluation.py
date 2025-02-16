import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure Google API
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure Google API
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Ground truth data based on the PDF content
ground_truth = {
    "What is Sachin Yadav's profession?": "Tutor",
    "What are Sachin Yadav's language skills?": "Native Nepali speaker and also knows English.",
    "What are Sachin Yadav's hobbies?": "Playing Cricket, Dancing, Reading, and Watching movies.",
    "What are Sachin Yadav's contact details?": "Phone - 9863756954, Email - yadsac2002@gmail.com, Address - Kirtipur, Kathmandu.",
    "What skills does Sachin Yadav have?": "AutoCAD, Microsoft Office, and slide presentations."
}

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input
def user_input(user_question, text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = get_vector_store(text_chunks)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Function to evaluate accuracy
def evaluate_accuracy(questions, ground_truth, model_responses):
    correct = 0
    for question in questions:
        if question in ground_truth:
            
            # Compute cosine similarity between ground truth and model response
            gt_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001").embed_query(ground_truth[question])
            response_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001").embed_query(model_responses[question])
            similarity = cosine_similarity([gt_embedding], [response_embedding])[0][0]

            # Consider correct if similarity is above a threshold (e.g., 0.8)
            if similarity > 0.8:
                correct += 1

    accuracy = (correct / len(questions)) * 100
    return accuracy

# Main function
def main():
    # Load PDFs
    pdf_docs = ["sachin-yadav.pdf"]  
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)

    # Define questions
    questions = list(ground_truth.keys())

    # Get model responses
    model_responses = {}
    for question in questions:
        response = user_input(question, text_chunks)
        model_responses[question] = response
        print(f"Question: {question}\nResponse: {response}\n")

    # Evaluate accuracy
    accuracy = evaluate_accuracy(questions, ground_truth, model_responses)
    print(f"Accuracy: {accuracy:.2f}%")

# Run the app
if __name__ == "__main__":
    main()