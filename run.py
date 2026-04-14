"""開発サーバー起動スクリプト

使い方:
  python run.py

環境変数:
  OPENAI_API_KEY: OpenAI API キー (必須)
  .env に設定しても読み込まれます。

起動後、ブラウザで http://localhost:8000 を開いてください。
"""

from dotenv import load_dotenv
import uvicorn

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
