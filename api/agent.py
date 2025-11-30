from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # load OPENAI_API_KEY


def run_agent(prompt: str) -> str:
    """Simple direct LLM call — stable and compatible."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing.")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=api_key
    )

    response = llm.invoke(prompt)

    return response.content
