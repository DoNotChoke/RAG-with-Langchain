import os
import re
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def extract_answer(
        text_response: str,
        pattern: str = r"Answer:\s*(.*)"
) -> str:
    match = re.match(pattern, text_response, re.DOTALL)
    if match:
        answer_text = match.group(1).strip()
        return answer_text
    else:
        return text_response


class Str_OutputParser(StrOutputParser):
    def __init__(self):
        super().__init__()

    def parse(self, text: str) -> str:
        return extract_answer(text)


class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = hub.pull("rlm/rag-prompt", api_key=os.getenv("LANGCHAIN_API_KEY"))
        self.str_parser = Str_OutputParser()

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_chain(self, retriever):
        input_data = {
            "context": retriever | self.format_docs,
            "question": RunnablePassthrough()
        }
        rag_chain = (
            input_data |
            self.prompt |
            self.llm |
            self.str_parser
        )
        return rag_chain

