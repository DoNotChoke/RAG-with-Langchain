from langchain_together import ChatTogether
import os
from dotenv import load_dotenv

load_dotenv()

def get_model(model_name: str="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
    return ChatTogether(model=model_name, api_key=os.getenv("TOGETHER_API_KEY"))