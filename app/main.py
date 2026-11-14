import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
#print(sys.path)
from typing import Any

from fastapi import FastAPI, Request, APIRouter, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app import __version__
from app.Hackathon_setup import face_recognition, exp_recognition

import io
import os
import numpy as np
from PIL import Image


app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# To store files uploaded by users
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# To access Templates directory
templates = Jinja2Templates(directory="app/templates")

simi_filename1 = None
simi_filename2 = None
face_rec_filename = None
expr_rec_filename = None


async def read_upload(upload: UploadFile, prefix: str = ''):
    """Read an uploaded file WITHOUT trusting content_type (mobile/camera uploads often send
    'application/octet-stream'). Returns (saved_path, RGB numpy array) or (None, None) if not a readable image."""
    contents = await upload.read()
    if not contents:
        return None, None
    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')   # also handles PNG (4-ch) / grayscale
    except Exception:
        return None, None                                        # e.g. HEIC or a non-image file
    name = os.path.splitext(os.path.basename(upload.filename or ''))[0] or 'upload'
    path = 'app/static/' + prefix + name + '.jpg'               # prefix keeps the 2 similarity images apart
    img.save(path, format='JPEG')                                # always a browser-viewable file
    return path, np.array(img).astype(np.uint8)

INVALID_IMAGE_MSG = "Could not read the image. Please upload a JPG or PNG photo."


####################################  Home Page endpoints  #################################################
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {'request': request,})


####################################  Face Similarity endpoints  #################################################
@app.get("/similarity/")
async def similarity_root(request: Request):
    return templates.TemplateResponse("similarity.html", {'request': request,})


@app.post("/predict_similarity/")
async def create_upload_files(request: Request, file1: UploadFile = File(...), file2: UploadFile = File(...)):
    global simi_filename1
    global simi_filename2
    
    simi_filename1, img1 = await read_upload(file1, 'img1_')
    simi_filename2, img2 = await read_upload(file2, 'img2_')
    if img1 is None or img2 is None:
        return templates.TemplateResponse("predict_similarity.html", {"request": request, "result": INVALID_IMAGE_MSG,
                                                       "simi_filename1": '', "simi_filename2": ''})
    
    result = face_recognition.get_similarity(img1, img2)
    #print(result)
    
    return templates.TemplateResponse("predict_similarity.html", {"request": request,
                                                       "result": np.round(result, 3),
                                                       "simi_filename1": '../static/'+os.path.basename(simi_filename1),
                                                       "simi_filename2": '../static/'+os.path.basename(simi_filename2),})
    

####################################  Face Recognition endpoints  #################################################
@app.get("/face_recognition/")
async def face_recognition_root(request: Request):
    return templates.TemplateResponse("face_recognition.html", {'request': request,})


@app.post("/predict_face_recognition/")
async def create_upload_files(request: Request, file3: UploadFile = File(...)):
    global face_rec_filename
    
    face_rec_filename, img1 = await read_upload(file3)
    if img1 is None:
        return templates.TemplateResponse("predict_face_recognition.html", {"request": request, "result": INVALID_IMAGE_MSG,
                                                       "face_rec_filename": ''})
        
    result = face_recognition.get_face_class(img1)
    print(result)
    
    return templates.TemplateResponse("predict_face_recognition.html", {"request": request,
                                                       "result": result,
                                                       "face_rec_filename": '../static/'+os.path.basename(face_rec_filename),})


####################################  Expresion Recognition endpoints  #################################################
@app.get("/expr_recognition/")
async def expr_recognition_root(request: Request):
    return templates.TemplateResponse("expr_recognition.html", {'request': request,})


@app.post("/predict_expr_recognition/")
async def create_upload_files(request: Request, file4: UploadFile = File(...)):
    global expr_rec_filename
    
    expr_rec_filename, img1 = await read_upload(file4)
    if img1 is None:
        return templates.TemplateResponse("predict_expr_recognition.html", {"request": request, "result": INVALID_IMAGE_MSG,
                                                       "expr_rec_filename": ''})
        
    result = exp_recognition.get_expression(img1)
    print(result)
    
    return templates.TemplateResponse("predict_expr_recognition.html", {"request": request,
                                                       "result": result,
                                                       "expr_rec_filename": '../static/'+os.path.basename(expr_rec_filename),})



# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Start app
if __name__ == "__main__":
    import os
    import uvicorn
    # Render/AWS set PORT automatically; local Docker defaults to 8001
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
