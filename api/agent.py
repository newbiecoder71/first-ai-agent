from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
import os

load_dotenv()  # for local dev

def run_agent(prompt: str) -> str:
    """Run a simple ReAct agent with no tools."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing.")

    # Basic LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=api_key
    )

    # Create a simple ReAct agent (zero tools)
    agent = initialize_agent(
        tools=[], 
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )

    # Run it
    result = agent.run(prompt)
    return result
