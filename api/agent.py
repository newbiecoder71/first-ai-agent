from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os

load_dotenv()  # load OPENAI_API_KEY

def run_agent(prompt: str) -> str:
    """Runs the AI agent and returns its output."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment variables.")

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # Simple agent (no tools yet)
    agent = create_react_agent(llm)

    # Run agent
    result = agent.invoke({"input": prompt})
    return result["output"]
