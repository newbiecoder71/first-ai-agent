from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel

# Import our agent
from agent import run_agent

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Setup FastAPI
app = FastAPI()

# Mount static directory (public folder)
app.mount("/public", StaticFiles(directory=BASE_DIR / "public"), name="public")

# Templates directory
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# Request/Response Models
class AgentRequest(BaseModel):
    prompt: str

class AgentResponse(BaseModel):
    response: str


# Routes
@app.get("/")
async def home(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/agent", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest):
    """Invoke the AI agent."""
    try:
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

        result = run_agent(request.prompt)
        return AgentResponse(response=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invoking agent: {str(e)}")
