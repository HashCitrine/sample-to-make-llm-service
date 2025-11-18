import os

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.llms import VLLM

from vector_store import init_vector_store


def init_qa_chain():
    llm = os.getenv("LLM_MODEL")
    vector_store = init_vector_store()

    llm = VLLM(
        model=llm,
        trust_remote_code=True,
        max_new_tokens=256,
        temperature=0.7,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)

    return qa_chain


if __name__ == "__main__":
    qa_chain = init_qa_chain()
    while True:
        query = input("질문을 입력하세요 (exit 입력 시 종료): ")
        if query.lower() == "exit":
            print("종료합니다.")
            break
        answer = qa_chain.run(query)
        print("답변:", answer)
