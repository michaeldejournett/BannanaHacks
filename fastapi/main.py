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


@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts."""
    try:
        from utils.model_loader import load_model
        print("Loading banana detection model...")
        load_model()
        print("✓ Model loaded successfully!")
    except ImportError as e:
        print(f"⚠ ERROR: Failed to import PyTorch: {e}")
        print("\n" + "="*60)
        print("SOLUTION: Install Microsoft Visual C++ Redistributable")
        print("Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("="*60)
        print("\nThe API will still run, but predictions will fail.")
    except Exception as e:
        print(f"⚠ Warning: Failed to load model: {e}")
        print("The API will still run, but predictions will fail.")


@app.get("/")
async def root():
    return {"message": "BannanaHacks API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}

