from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
from pathlib import Path
from langchain_decorators import chain
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)


llm = OpenAI(temperature=0.7)


env_path = Path('.') / '.env'
load_dotenv(env_path)