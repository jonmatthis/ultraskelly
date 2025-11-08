import logging
logger = logging.getLogger(__name__)

from pathlib import Path

from fastapi import HTTPException, APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

audio_router = APIRouter(tags=["Audio"], prefix="/audio")

# const formData = new FormData();
#
# // Convert ArrayBuffer to Blob
# const audioBlob = new
# Blob([request.audioData], {type: 'audio/pcm'});
#
# formData.append('audio', audioBlob, `${request.itemId}.pcm`);
# formData.append('itemId', request.itemId);
# formData.append('timestamp', request.timestamp);
# formData.append('metadata', JSON.stringify(request.metadata | | {}));

class AudioUploadRequest(BaseModel):
    itemId: str
    timestamp: int
    metadata: dict | None = None
    audioData: bytes

class AudioUploadResponse(BaseModel):
    success: bool
    message: str | None = None

@audio_router.post("/upload", response_model=AudioUploadResponse)
async def upload_audio(request: AudioUploadRequest) -> AudioUploadResponse:
    """Endpoint to upload audio data"""
    logger.info(f"Received audio upload request for itemId: {request.itemId}")

    audio_dir = Path("uploaded_audios")
    audio_dir.mkdir(exist_ok=True, parents=True)

    audio_file_path = audio_dir / f"{request.itemId}.pcm"

    try:
        with open(audio_file_path, "wb") as audio_file:
            audio_file.write(request.audioData)
        logger.info(f"Audio data saved to {audio_file_path}")

        return AudioUploadResponse(success=True, message="Audio uploaded successfully")
    except Exception as e:
        logger.error(f"Error saving audio data: {e}")
        raise HTTPException(status_code=500, detail="Failed to save audio data")