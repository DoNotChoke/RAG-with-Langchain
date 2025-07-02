from pydantic import BaseModel, Field

from src.rag.file_loader import Loader
from src.rag.vectorstore import VectorDB
from src.rag.offline_rag import Offline_RAG
from src.llm_model import get_model

class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")

class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer to the question")

def build_rag_chain(llm, data_dir, data_type):
    doc_loaded = Loader(file_type=data_type).load_dir(data_dir, workers=2)
    print("Create vector db")
    retriever = VectorDB(documents=doc_loaded).get_retriever()
    print("Create model")
    rag_chain = Offline_RAG(llm=llm).get_chain(retriever)
    return rag_chain

if __name__ == '__main__':
    model = get_model()
    build_rag_chain(model, data_dir="D:\RAGWithLangchain\src\data", data_type="pdf")