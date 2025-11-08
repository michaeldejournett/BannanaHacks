from fastapi import FastAPI
from fastapi.responses import Response

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests to prevent 404 errors in browser console."""
    return Response(status_code=204)
