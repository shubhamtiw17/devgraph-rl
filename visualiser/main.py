from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from visualiser.routers.graphs import router as graph_router
from visualiser.routers.memory import router as memory_router
from visualiser.routers.repo import router as repo_router

app = FastAPI(title="DevGraph-RL Visualiser", version="1.0.0")

app.include_router(graph_router, prefix="/api")
app.include_router(memory_router)
app.include_router(repo_router)

app.mount("/static", StaticFiles(directory="visualiser/static"), name="static")

@app.get("/")
async def root():
    return FileResponse("visualiser/static/index.html")