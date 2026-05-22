"""FastAPI web server for DALAL TERMINAL.

Serves the terminal-styled single-page frontend, dispatches command-bar
input via POST /api/command, and streams the live ticker tape over a
websocket. The command dispatch runs in a worker thread so the tape
stays smooth while a heavy command (e.g. TOP) is computed.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import universe
from config.settings import get_settings
from monitoring.logging import get_logger, setup_logging
from ui.commands import CommandRouter

WEB_DIR = Path(__file__).parent / "web"

settings = get_settings()
setup_logging(settings.log_level)
log = get_logger("ui.server")
router = CommandRouter(settings)

_clients: set[WebSocket] = set()


class CommandIn(BaseModel):
    command: str


async def _tape_loop() -> None:
    """Advance demo prices and broadcast the ticker tape every ~1.5s."""
    while True:
        try:
            router.quotes.step()
            tape = []
            for sym in universe.symbols():
                q = router.quotes.snapshot(sym)
                if q is not None:
                    tape.append({"s": sym, "ltp": round(q.ltp, 2),
                                 "chg": round(q.change_pct, 2), "d": q.direction})
            msg = json.dumps({"type": "tape", "data": tape})
            for client in list(_clients):
                try:
                    await client.send_text(msg)
                except Exception:
                    _clients.discard(client)
        except Exception:  # never let the background loop die
            log.exception("tape loop error")
        await asyncio.sleep(1.5)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info("DALAL TERMINAL starting (mode=%s)", settings.mode)
    task = asyncio.create_task(_tape_loop())
    yield
    task.cancel()


app = FastAPI(title="DALAL TERMINAL", lifespan=lifespan)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "mode": settings.mode,
            "universe": len(universe.symbols())}


@app.post("/api/command")
async def command(payload: CommandIn) -> dict:
    # dispatch off the event loop - TOP/SCAN can take a moment
    return await asyncio.to_thread(router.dispatch, payload.command)


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    _clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # keepalive pings, ignored
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(websocket)


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
