"""Single-process public demo: Dash at / and FastAPI routes on the same port."""

import json
import logging
import os
from pathlib import Path

import uvicorn
from a2wsgi import WSGIMiddleware
from fastapi import FastAPI
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.api.main import create_app as create_api
from src.data.dashboard import DemoMetadata
from src.data.loader import PROJECT_ROOT
from src.frontend.app import create_app as create_dashboard
from src.frontend.client import PortfolioClient


logger = logging.getLogger(__name__)


class RequestBodyLimit:
    def __init__(self, app: ASGIApp, max_bytes: int = 2 * 1024 * 1024) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_bytes:
                logger.warning("Public demo request exceeded body limit: %s", scope["path"])
                response = JSONResponse(
                    {"detail": "Public demo requests are limited to 2 MiB."},
                    status_code=413,
                )
                await response(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": b"".join(chunks), "more_body": False}

        await self.app(scope, replay, send)


def create_app(*, bundle_directory: Path, port: int) -> FastAPI:
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535.")
    manifest = json.loads(
        (bundle_directory / "portfolio" / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest["schema_version"] != 2 or not 3 <= manifest["rows"] <= 10_000:
        raise ValueError("Public demo requires a labelled demo bundle of at most 10,000 rows.")
    demo = DemoMetadata.model_validate(manifest["demo"])
    notice = (
        f"PUBLIC DEMO - {manifest['rows']:,} sampled synthetic customers from "
        f"{demo.source_rows:,}. Same trained model; original training thresholds. "
        "Portfolio totals and campaign capacity apply only to this demo, not the full "
        "dataset. No model evaluation is performed here. Do not enter real personal data."
    )
    api = create_api(bundle_directory=bundle_directory)
    api.add_middleware(RequestBodyLimit)
    dashboard = create_dashboard(
        PortfolioClient(f"http://127.0.0.1:{port}"), demo_notice=notice,
    )
    api.mount("/", WSGIMiddleware(dashboard.server, workers=4))
    return api


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("PORT", "8000"))
    directory = Path(os.environ.get(
        "CHURN_BUNDLE_DIRECTORY", str(PROJECT_ROOT / "deploy" / "demo_bundle"),
    ))
    app = create_app(bundle_directory=directory, port=port)
    uvicorn.run(
        app, host="0.0.0.0", port=port, workers=1,
        limit_concurrency=24, timeout_keep_alive=5, access_log=False,
    )


if __name__ == "__main__":
    main()
