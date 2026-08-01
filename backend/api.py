from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import io
import os
import tempfile
from PIL import Image

from utils.model_manager import ModelManager
from backend.exporter import export_zip

app = FastAPI(
    title="IRIS-AI API",
    description="Inference API backend for the Intelligent Remote-sensing Infrared Interpretation Suite",
    version="1.0.0"
)
mm = ModelManager.get_instance()

@app.on_event("startup")
async def startup_event():
    """Preload all pipeline deep learning models on server startup."""
    mm.preload_all()

@app.get("/")
def read_root():
    """Root health check endpoint."""
    return {"status": "ok", "message": "IRIS-AI Backend Running"}

@app.get("/status")
def get_status():
    """Server status endpoint."""
    return {"status": "ok", "message": "IRIS-AI Backend Running"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    run_enhancement: bool = True,
    run_superres: bool = True,
    run_colorize: bool = True,
    run_segmentation: bool = True,
    run_caption: bool = True
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        results = mm.run(
            image_input=image,
            run_enhancement=run_enhancement,
            run_superres=run_superres,
            run_colorize=run_colorize,
            run_segmentation=run_segmentation,
            run_caption=run_caption,
            source_filename=file.filename
        )
        
        # Export everything as a ZIP and return it
        zip_bytes = export_zip(results)
        
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "results.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)
            
        return FileResponse(
            zip_path, 
            media_type="application/zip", 
            filename=f"IRIS_AI_{file.filename}.zip"
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
