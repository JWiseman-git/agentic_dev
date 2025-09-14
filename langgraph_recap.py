from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from rich.markdown import Markdown
from langchain.tools import tool

load_dotenv()

llm = init_chat_model("gpt-4o-mini")

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """PLACEHOLDER: Write and send an email."""
    return f"Email sent to {to} with subject '{subject}' and content: {content}"

model_with_tools = llm.bind_tools([write_email], tool_choice="any", parallel_tool_calls=False)

#Arguments inferred from Natural Lang

output = model_with_tools.invoke("Draft a response to my boss (boss@company.ai) about tomorrow's meeting")

args = output.tool_calls[0]['args']
print(args)
