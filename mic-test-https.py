from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import os
import socket
from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None
    ):
        super().__init__(app)
        self.host = host
        self.port = port

    async def dispatch(self, request: Request, call_next):
        # Check if the request is HTTP
        if request.url.scheme == "http":
            # Construct HTTPS URL
            host = self.host or request.headers.get("host").split(":")[0]
            port = f":{self.port}" if self.port else ""
            url = request.url.copy_with(scheme="https", netloc=f"{host}{port}")
            return RedirectResponse(url=str(url), status_code=307)
        return await call_next(request)

# Get local IP address
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "localhost"

app = FastAPI()
local_ip = get_local_ip()
HTTP_PORT = 8000
HTTPS_PORT = 8443  # Separate port for HTTPS

# Add HTTPS redirect middleware
app.add_middleware(
    HTTPSRedirectMiddleware,
    host=local_ip,
    port=HTTPS_PORT
)

# Configure CORS with specific origins
ALLOWED_ORIGINS = [
    f"https://{local_ip}:{HTTPS_PORT}",
    f"https://localhost:{HTTPS_PORT}",
    f"https://127.0.0.1:{HTTPS_PORT}"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Permissions-Policy"] = "microphone=*, camera=()"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    origin = request.headers.get("origin")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    
    return response

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    with open("static/index.html") as f:
        content = f.read()
        # Replace the hardcoded localhost in the HTML with the server's IP and HTTPS
        content = content.replace(
            'fetch(\'http://localhost:8000/upload\'',
            f'fetch(\'https://{local_ip}:{HTTPS_PORT}/upload\''
        )
        return Response(
            content=content,
            media_type="text/html",
            headers={
                "Permissions-Policy": "microphone=*, camera=()",
                "Cross-Origin-Opener-Policy": "same-origin",
                "Cross-Origin-Embedder-Policy": "require-corp",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
            }
        )

@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...), device_id: str = None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}_{device_id}.webm"
    file_path = os.path.join("uploads", filename)
    
    with open(file_path, "wb") as f:
        content = await audio.read()
        f.write(content)
    
    return {
        "filename": filename,
        "path": file_path,
        "size": len(content),
        "device_id": device_id
    }

def run_server():
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=HTTPS_PORT,
        ssl_keyfile="key.pem",
        ssl_certfile="cert.pem"
    )
    server = uvicorn.Server(config)
    
    print(f"\nServer running on:")
    print(f"HTTPS: https://localhost:{HTTPS_PORT}")
    print(f"Network HTTPS: https://{local_ip}:{HTTPS_PORT}")
    print(f"All HTTP traffic will be redirected to HTTPS\n")
    
    server.run()

if __name__ == "__main__":
    run_server()