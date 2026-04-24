"""Development server launcher.

Usage:
  python run.py

Environment:
  OPENAI_API_KEY: OpenAI API key. Values in .env are loaded automatically.

After startup, open http://localhost:8000 in your browser.
"""

from dotenv import load_dotenv
import uvicorn

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host="localhost",
        port=8000,
        reload=True,
    )
