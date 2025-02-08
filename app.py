import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

# Store every pdf pages into single file
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

# Getting chunks of the text
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator= '\n',
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )

    chunks = text_splitter.split_text(raw_text)
    return chunks

def main():
    load_dotenv()
    st.set_page_config(page_title="Multiple PDFs chat",page_icon=":books:")

    st.header("Chat with multiple PDFs :books:")
    st.text_input("Ask question about your doc")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                
                # Store all the pdf pages into single variable
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # get text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # create vector store


if __name__ == '__main__':
    main()