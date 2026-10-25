import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.serving.router import router
from src.serving.model_loader import load_model

app = FastAPI(
    title="LendingClub Default Risk Prediction Service",
    description="FastAPI service for scoring default risk and detecting feature drift.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Register router
app.include_router(router)

@app.on_event("startup")
def startup_event():
    print("FastAPI application startup: Loading model...")
    try:
        load_model()
    except Exception as e:
        print(f"Error during startup model load: {e}")
        # Note: Do not raise here so the health endpoint can still report degraded status

if __name__ == "__main__":
    # Get port from environment or use default 8080
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("src.serving.app:app", host="0.0.0.0", port=port, reload=False)
