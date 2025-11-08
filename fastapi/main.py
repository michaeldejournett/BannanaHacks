from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import banana_analysis

app = FastAPI(title="BannanaHacks API", version="1.0.0")

# Configure CORS to allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port and common React port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(banana_analysis.router, prefix="/api", tags=["banana"])


@app.get("/")
async def root():
    return {"message": "BannanaHacks API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}

