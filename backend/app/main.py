from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import config
from .db import init_db
from .routes import ai, annotations, events, export, images, projects, stats


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="AI Annotator API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

for module in (projects, images, annotations, ai, events, stats, export):
    app.include_router(module.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


class UIFiles(StaticFiles):
    """Static UI where index.html is always revalidated.

    Assets have content-hashed names, but index.html does not: without this,
    browsers (notably mobile Chrome) keep a stale index.html after a rebuild and
    load the old bundle.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        if response.media_type == "text/html":
            response.headers["Cache-Control"] = "no-cache"
        return response


# Mounted last so /api routes take precedence. Not present in the docker api
# image (nginx serves the UI there).
if config.FRONTEND_DIST.is_dir():
    app.mount("/", UIFiles(directory=config.FRONTEND_DIST, html=True), name="ui")
