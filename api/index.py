import sys
import os
import traceback
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Force adding the current directory to sys.path to fix import issues
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    # Try absolute import first, then relative compatible fallback
    try:
        from server import app
    except ImportError:
        from .server import app
except Exception as e:
    # Capture the error
    error_details = traceback.format_exc()
    
    # Create a fallback app to serve the error in the browser
    app = FastAPI()
    
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
    async def catch_all(path: str):
        return JSONResponse(
            status_code=200, 
            content={
                "error": "Critical Startup Error",
                "message": str(e),
                "traceback": error_details,
                "python_version": sys.version,
                "current_dir": current_dir,
                "sys_path": sys.path
            }
        )
