"""
Banana Analysis Controller (lightweight).
This version removes PyTorch dependencies and returns a simple acknowledgement
when an image is uploaded. It's useful for development and testing when you
don't want to install heavy native libraries.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/banana", tags=["banana"])


@router.post("/analyze")
async def analyze_banana(file: UploadFile = File(...)):
    """Accept an image upload and return a simple acknowledgement.

    This intentionally avoids importing PyTorch or other heavy libraries so
    the FastAPI app can start in environments where torch is not installed.
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPEG, JPG, or PNG image."
        )

    # Read the file (we don't process it here)
    contents = await file.read()
    size = len(contents)

    return JSONResponse(content={
        "message": "image received",
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": size
    })


@router.get("/health")
async def health_check():
    """Lightweight health check that doesn't require torch."""
    return {"status": "healthy", "model_loaded": False, "message": "PyTorch disabled for this dev build"}
