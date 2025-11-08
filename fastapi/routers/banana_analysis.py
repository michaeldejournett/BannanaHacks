from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

router = APIRouter()

# Allowed image MIME types
ALLOWED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg"
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {".png", ".jpeg", ".jpg"}


def is_valid_image_file(file: UploadFile) -> bool:
    """
    Validate that the uploaded file is a PNG, JPEG, or JPG image.
    
    Args:
        file: The uploaded file
        
    Returns:
        True if the file is a valid image type, False otherwise
    """
    # Check MIME type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        return False
    
    # Check file extension as additional validation
    filename = file.filename.lower() if file.filename else ""
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return False
    
    return True


@router.post("/banana-analysis")
async def analyze_banana(file: UploadFile = File(...)):
    """
    Upload a banana image for analysis.
    
    Accepts PNG, JPEG, or JPG image files.
    
    Args:
        file: The image file to upload
        
    Returns:
        Success message if file is valid, error message otherwise
    """
    # Validate file type
    if not is_valid_image_file(file):
        return {"message": "error"}
    
    # File is valid - return success
    # Note: The actual file processing/scanning will be handled elsewhere
    return {"message": "successful upload!"}

