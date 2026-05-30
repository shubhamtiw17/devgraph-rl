from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path('/home/shubham/devgraph-rl/.env'))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from visualiser.routers.graphs import router as graph_router
from visualiser.routers.memory import router as memory_router
from visualiser.routers.repo import router as repo_router
from visualiser.routers.sandbox import router as sandbox_router
from visualiser.routers.rewards import router as rewards_router
from visualiser.routers.training import router as training_router
from visualiser.routers.assistant import router as assistant_router

app = FastAPI(title="DevGraph-RL Visualiser", version="1.0.0")
app.include_router(graph_router, prefix="/api")
app.include_router(memory_router)
app.include_router(repo_router)
app.include_router(sandbox_router)
app.include_router(rewards_router)
app.include_router(training_router)
app.include_router(assistant_router)
app.include_router(assistant_router)

app.mount("/static", StaticFiles(directory="visualiser/static"), name="static")

@app.get("/")
async def root():
    return FileResponse("visualiser/static/index.html")