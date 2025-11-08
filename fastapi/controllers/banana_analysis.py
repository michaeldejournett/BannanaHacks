from fastapi import File, UploadFile
from fastapi.routing import APIRouter

router = APIRouter()

@router.post("/bananaAnalysis")
async def bananaAnalysis(file: UploadFile = File(...)):
    """
    Controller for analyzing banana images.
    Accepts PNG or JPEG images as input.
    """
    # Validate file type
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        return {"error": "Invalid file type. Please upload a PNG or JPEG image."}
    
    # Read the file content
    contents = await file.read()
    
    # Process the image here (placeholder for now)
    # You can add your image analysis logic here
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(contents),
        "message": "Image received successfully"
    }

