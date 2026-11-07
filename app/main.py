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
    
    if 'image' in file1.content_type:
        contents = await file1.read()
        simi_filename1 = 'app/static/' + file1.filename
        with open(simi_filename1, 'wb') as f:
            f.write(contents)
    
    if 'image' in file2.content_type:
        contents = await file2.read()
        simi_filename2 = 'app/static/' + file2.filename
        with open(simi_filename2, 'wb') as f:
            f.write(contents)
    
    # .convert('RGB') handles PNG (4-channel) and grayscale uploads that broke the old reshape
    img1 = np.array(Image.open(simi_filename1).convert('RGB')).astype(np.uint8)
    img2 = np.array(Image.open(simi_filename2).convert('RGB')).astype(np.uint8)
    
    result = face_recognition.get_similarity(img1, img2)
    #print(result)
    
    return templates.TemplateResponse("predict_similarity.html", {"request": request,
                                                       "result": np.round(result, 3),
                                                       "simi_filename1": '../static/'+file1.filename,
                                                       "simi_filename2": '../static/'+file2.filename,})
    

####################################  Face Recognition endpoints  #################################################
@app.get("/face_recognition/")
async def face_recognition_root(request: Request):
    return templates.TemplateResponse("face_recognition.html", {'request': request,})


@app.post("/predict_face_recognition/")
async def create_upload_files(request: Request, file3: UploadFile = File(...)):
    global face_rec_filename
    
    if 'image' in file3.content_type:
        contents = await file3.read()
        face_rec_filename = 'app/static/' + file3.filename
        with open(face_rec_filename, 'wb') as f:
            f.write(contents)
    
    img1 = np.array(Image.open(face_rec_filename).convert('RGB')).astype(np.uint8)
        
    result = face_recognition.get_face_class(img1)
    print(result)
    
    return templates.TemplateResponse("predict_face_recognition.html", {"request": request,
                                                       "result": result,
                                                       "face_rec_filename": '../static/'+file3.filename,})


####################################  Expresion Recognition endpoints  #################################################
@app.get("/expr_recognition/")
async def expr_recognition_root(request: Request):
    return templates.TemplateResponse("expr_recognition.html", {'request': request,})


@app.post("/predict_expr_recognition/")
async def create_upload_files(request: Request, file4: UploadFile = File(...)):
    global expr_rec_filename
    
    if 'image' in file4.content_type:
        contents = await file4.read()
        expr_rec_filename = 'app/static/' + file4.filename
        with open(expr_rec_filename, 'wb') as f:
            f.write(contents)
    
    img1 = np.array(Image.open(expr_rec_filename).convert('RGB')).astype(np.uint8)
        
    result = exp_recognition.get_expression(img1)
    print(result)
    
    return templates.TemplateResponse("predict_expr_recognition.html", {"request": request,
                                                       "result": result,
                                                       "expr_rec_filename": '../static/'+file4.filename,})



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
