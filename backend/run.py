"""開発サーバー起動スクリプト

使い方:
  python run.py

環境変数 (Issue #17):
  本番運用はローカル推論バックエンド (vLLM 等) を既定とし、OpenAI API キーは不要。

  LLM_BACKEND:   "local" (既定) | "openai"
  LLM_ENDPOINT:  ローカル推論サーバ URL (例: http://localhost:8001/v1)
  LLM_MODEL:     推論サーバが配信するモデル名 (例: qwen3.6-35b-nvfp4)
  OPENAI_API_KEY: optional。蒸留・ベンチマーク用途で OpenAI を叩く場合のみ。

  詳細は backend/.env.example と backend/config.yaml を参照。

起動後、ブラウザで http://localhost:8000 を開いてください。
"""

import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host="localhost",
        port=8000,
        reload=True,
    )
