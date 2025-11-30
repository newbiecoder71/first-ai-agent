from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

# Import run_agent from api/agent.py
from api.agent import run_agent

# BASE DIRECTORY OF PROJECT ROOT (not /api)
BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI()

# Mount /public → public folder
app.mount("/public", StaticFiles(directory=BASE_DIR / "public"), name="public")

# Setup templates directory
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# Request model
class AgentRequest(BaseModel):
    prompt: str


# Response model
class AgentResponse(BaseModel):
    response: str


@app.get("/")
async def home(request: Request):
    """
    Serve the homepage.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/agent", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest):
    """
    Call the AI agent with user input.
    """
    try:
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

        result = run_agent(request.prompt)
        return AgentResponse(response=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invoking agent: {str(e)}")
