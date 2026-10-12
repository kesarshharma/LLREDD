"""FastAPI Application Entry Point."""

import argparse
from pathlib import Path
import sys

# `python api/main.py` puts `api/`, rather than the project root, at sys.path[0].
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="LEREDD API",
    description="LLM-Enabled REquirement Dependency Detection REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "message": "Welcome to LEREDD API",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


def main(argv=None) -> int:
    """Run the local API server; the installed CLI exposes the same operation."""
    parser = argparse.ArgumentParser(description="Run the LEREDD FastAPI server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")

    import uvicorn

    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
