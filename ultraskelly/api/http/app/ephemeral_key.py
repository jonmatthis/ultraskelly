import loggingimport os
from datetime import datetime

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException, APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)
load_dotenv()

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.error("❌ ERROR: OPENAI_API_KEY is not set in .env file")
    logger.error("Please create a .env file and add your OpenAI API key")
    exit(1)

logger.info("✅ OpenAI API key loaded successfully")

ephemeral_key_router = APIRouter(tags=["App"])

class EphemeralKeyResponse(BaseModel):
    value: str
    expires_at: int


@ephemeral_key_router.post("/ephemeral-key", response_model=EphemeralKeyResponse)
async def get_ephemeral_key() -> EphemeralKeyResponse:
    """Generate a temporary ephemeral key for OpenAI Realtime API"""
    print("🔑 Generating new ephemeral key...")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/realtime/client_secrets",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "session": {
                        "type": "realtime",
                        "model": "gpt-realtime"
                    }
                },
                timeout=30.0
            )

            if response.status_code != 200:
                error_text = response.text
                print(f"❌ OpenAI API error: {error_text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to generate ephemeral key: {error_text}"
                )

            data = response.json()
            print("✅ Ephemeral key generated successfully")

            return EphemeralKeyResponse(
                value=data["value"],
                expires_at=data.get("expires_at", int(datetime.now().timestamp() * 1000) + 60000)
            )

        except httpx.HTTPError as e:
            print(f"❌ HTTP error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to connect to OpenAI API: {str(e)}"
            )
