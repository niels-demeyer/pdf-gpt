import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

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

load_dotenv()


def main():
    st.header("Chat with PDF 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f"{store_name}")
        # st.write(chunks)

        embeddings = OpenAIEmbeddings()

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                st.write("Loading VectorStore from disk...")
                VectorStore_bytes = pickle.load(f)
                VectorStore = FAISS.deserialize_from_bytes(
                    VectorStore_bytes, embeddings
                )
                st.write("VectorStore loaded.")
        else:
            st.write("VectorStore not found on disk. Creating new one...")
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                st.write("Saving VectorStore to disk...")
                VectorStore_bytes = VectorStore.serialize_to_bytes()
                pickle.dump(VectorStore_bytes, f)
                st.write("VectorStore saved.")

        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)


if __name__ == "__main__":
    main()
