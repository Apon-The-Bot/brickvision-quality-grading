import os
import shutil
import uuid
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import predict

app = FastAPI(title="Brick Grade Classification Server")

# Allow CORS for front-end requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model package
MODEL_PATH = "best_model_package.pkl"
MAT_PATH = "BrickSVMModel.mat"
model_package = None

@app.on_event("startup")
def startup_event():
    global model_package
    try:
        print(f"Loading Brick SVM model package from: {MODEL_PATH}...")
        model_package = predict.load_model(MODEL_PATH, MAT_PATH)
        print("Brick SVM Model package loaded successfully!")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model package. Error: {e}")
        import traceback
        traceback.print_exc()

@app.get("/")
def read_root():
    """Serves the main application user interface."""
    index_path = "index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend index.html file not found.")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Receives an uploaded brick image, extracts 35 features,
    compares them with the trained SVM classes, and returns the classification.
    """
    # 1. Verify file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    
    # Load model dynamically from disk on each request to prevent caching/out-of-sync issues
    try:
        current_model_package = predict.load_model(MODEL_PATH, MAT_PATH)
    except Exception as model_err:
        raise HTTPException(status_code=500, detail=f"Failed to load model package: {str(model_err)}")
        
    # 2. Create temp directory if it doesn't exist
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a unique filename
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    
    try:
        # 3. Save uploaded file to disk
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 4. Measure execution time
        start_time = time.time()
        
        # 5. Run feature extraction and prediction
        result = predict.predict_image(temp_file_path, current_model_package)
        
        end_time = time.time()
        analysis_time = round(end_time - start_time, 2)
        
        # Add analysis metrics
        result["analysis_time"] = analysis_time
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"Error during brick image analysis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")
        
    finally:
        # 6. Clean up temporary file from disk
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as clean_err:
                print(f"Failed to remove temp file {temp_file_path}: {clean_err}")

if __name__ == "__main__":
    import socket
    port = 8000
    # Check if port is in use by attempting to bind to it
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
        in_use = False
    except OSError:
        in_use = True
        
    if in_use:
        port = 8001
        print(f"Port 8000 is already in use (possibly by Sand Classification or TIME_WAIT state). Switching to port {port}...")
    
    # Start the server locally
    uvicorn.run(app, host="127.0.0.1", port=port)
