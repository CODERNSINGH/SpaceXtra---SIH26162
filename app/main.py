from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import BASE_DIR
from app.database import init_db
from app.pipeline.industrial_index import start_background_build_if_needed
from app.routes import pages, api

app = FastAPI(title="ThermoWatch AI")

init_db()
start_background_build_if_needed(log=print)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")

app.include_router(pages.router)
app.include_router(api.router, prefix="/api")
