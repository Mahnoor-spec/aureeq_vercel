import sys
import traceback
from fastapi import FastAPI
from fastapi.responses import JSONResponse

try:
    from .server import app
except Exception as e:
    # Capture the error
    error_details = traceback.format_exc()
    
    # Create a fallback app to serve the error
    app = FastAPI()
    
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
    async def catch_all(path: str):
        return JSONResponse(
            status_code=200, # Return 200 so Vercel doesn't mark it as crashed
            content={
                "error": "Critical Startup Error",
                "message": str(e),
                "traceback": error_details,
                "python_version": sys.version,
                "path": path
            }
        )
