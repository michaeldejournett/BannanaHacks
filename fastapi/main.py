from fastapi import FastAPI
from fastapi.responses import Response
from controllers.banana_analysis import router as banana_analysis_router

app = FastAPI()

# Include routers
app.include_router(banana_analysis_router)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests to prevent 404 errors in browser console."""
    return Response(status_code=204)
