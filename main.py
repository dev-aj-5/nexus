import os
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

app = FastAPI(title="Nexus Core Service")

@app.get("/")
@app.get("/health")
async def health_check():
    """Health check route targeted by ping services to keep the container warm."""
    return {"status": "operational", "system": "NEXUS_CHIEF_OF_STAFF"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="warning")