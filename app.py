from pathlib import Path
import traceback
import uvicorn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend import run_travel_graph

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="TripAI",
    description= "Multi Agent Travel Planner",
    version= "1.0.0"
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name= "static"
)

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)