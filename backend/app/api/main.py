"""FastAPI アプリケーションメイン"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.audio_routes import router as audio_router
from app.api.feedback import router as feedback_router
from app.api.routes import router
from app.scoring.weights import load_config
from app.store.db import init_db

logger = logging.getLogger(__name__)

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    app.state.config = config
    logger.info("設定読み込み完了: model=%s", config.llm_model)
    init_db()
    logger.info("フィードバック DB 初期化完了")
    yield


app = FastAPI(
    title="会議貢献度スコアリング",
    description="会議の文字起こしを分析し、各発言の貢献度を評価するAPI",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
app.include_router(audio_router, prefix="/api")
app.include_router(feedback_router, prefix="/api")

# 静的ファイル配信 (UI)
ui_dir = Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "dist"
if ui_dir.exists():
    app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")
else:
    logger.warning(
        "frontend/dist が見つかりません。UI を配信するには `cd frontend && npm run build` を実行してください。"
        " 開発時は Vite dev server (http://localhost:5173) を使用してください。"
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root_no_ui() -> HTMLResponse:
        return HTMLResponse(
            "<p>UI が未ビルドです。"
            "<code>cd frontend &amp;&amp; npm run build</code> を実行するか、"
            "開発時は <a href='http://localhost:5173'>http://localhost:5173</a> を使用してください。</p>"
        )
