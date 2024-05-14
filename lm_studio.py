import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForMaskedLM, pipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.globals import set_verbose, get_verbose
from openai import OpenAI
import os
import torch

load_dotenv()

# Instantiate the MLM pipeline
unmasker = pipeline("fill-mask", model="bert-base-multilingual-cased")

# Sidebar contents
with st.sidebar:
    st.title("🤗💬 LLM Chat A")
    st.markdown(
        """
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    - [documentation](https://github.com/niels-demeyer/pdf-gpt)
 
    """
    )
    add_vertical_space(5)
    st.write(
        "Make sure to connect your OpenAI API key to the app by following the documentation. (See the link above)"
    )

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


def main():
    st.header("Chat with PDF 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # embeddings
        store_name = pdf.name[:-4]
        st.write(f"{store_name}")

        # Define the model name and encoding arguments
        model_name = "bert-base-multilingual-cased"

        # Instantiate the BERT model
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            embeddings = []
            for chunk in chunks:
                embedding = (
                    client.embeddings.create(input=[chunk], model=model_name)
                    .data[0]
                    .embedding
                )
                embeddings.append(embedding)

            query_embedding = (
                client.embeddings.create(input=[query], model=model_name)
                .data[0]
                .embedding
            )

            # Compute cosine similarity between query and each document
            similarities = cosine_similarity([query_embedding], embeddings)
            # Get the indices of the top 3 most similar documents
            top_docs_indices = similarities.argsort()[-3:][::-1].flatten().tolist()
            top_docs = [chunks[i] for i in top_docs_indices]

            completion = client.chat.completions.create(
                model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query},
                ],
                temperature=0.7,
            )

            st.write(completion.choices[0].message)
        else:
            st.write("Please enter a query to chat with the embeddings.")


if __name__ == "__main__":
    main()
