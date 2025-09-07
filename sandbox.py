# Langchain docs step: https://python.langchain.com/docs/tutorials/llm_chain/

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

env_path = Path('.') / '.env'
load_dotenv(env_path)

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# messages = [
#     SystemMessage(content="Translate the following from English into Italian"),
#     HumanMessage(content="hi!"),
# ]

# system_template = "Translate the following from English into {language}"
#
# prompt_template = ChatPromptTemplate.from_messages(
#     [("system", system_template), ("user", "{text}")]
# )
# prompt = prompt_template.invoke({"language": "Italian", "text": "ciao"})
# print(prompt.to_messages())

from typing import Optional

from pydantic import BaseModel, Field

# Pydantic
class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


# structured_llm = llm.with_structured_output(Joke)
#
# structured_llm.invoke("Tell me a joke about cats")

#

from langchain_core.documents import Document

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]