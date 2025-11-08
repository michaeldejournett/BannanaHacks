from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from controllers.banana_analysis import router as banana_router

app = FastAPI(title="Banana Hacks API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(banana_router)

# Include routers
app.include_router(banana_analysis_router)

@app.get("/")
async def root():
    return {"message": "Banana Hacks API - Banana Ripeness Detection"}


@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests to prevent 404 errors in browser console."""
    return Response(status_code=204)
