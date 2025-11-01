import os
from datetime import datetime
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY is not set in .env file")
    print("Please create a .env file and add your OpenAI API key")
    exit(1)

print("✅ OpenAI API key loaded successfully")

app = FastAPI(title="Voice Agent Backend")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EphemeralKeyResponse(BaseModel):
    value: str
    expires_at: int


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    message: str


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint to verify server is running"""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        message="Backend is running!"
    )


@app.post("/api/ephemeral-key", response_model=EphemeralKeyResponse)
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


# Serve static files from the frontend build
# After you run `npm run build` in the frontend, this will serve those files
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Serve the frontend application"""
    frontend_path = "dist"
    file_path = os.path.join(frontend_path, full_path)
    
    # If file exists, serve it
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # Otherwise serve index.html (for client-side routing)
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    raise HTTPException(status_code=404, detail="Frontend not built yet. Run 'npm run build' first.")


if __name__ == "__main__":
    import uvicorn
    
    print("")
    print("🚀 Backend server starting...")
    print("📍 URL: http://localhost:3001")
    print("")
    print("Available endpoints:")
    print("  ✅ Health Check: http://localhost:3001/health")
    print("  🔑 Get Key:      http://localhost:3001/api/ephemeral-key")
    print("")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3001,
        log_level="info"
    )
