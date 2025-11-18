import os

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores import Chroma
from langchain_community.llms import VLLM

data_path = os.getenv("RAW_DATA_PATH")

embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)


def init_vector_store(persist_dir="./chroma_db"):
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        def get_chuck_docs(chunk_size=1000, chunk_overlap=100):
            loader = TextLoader(data_path)
            documents = loader.load()
            splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chuck_docs = splitter.split_documents(documents)

            return chuck_docs

        chuck_docs = get_chuck_docs()
        vector_store = Chroma.from_documents(chuck_docs, embedding_model, persist_directory=persist_dir)
        vector_store.persist()  # 디스크에 저장

    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

    return vector_store
