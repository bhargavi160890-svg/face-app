import numpy as np
import cv2
# MEMORY OPTION 1: matplotlib removed (not used on the server, saves ~20-40MB RAM)
import torch
# In the below line,remove '.' while working on your local system. However Make sure that '.' is present before face_recognition_model while uploading to the server, Do not remove it.
from .face_recognition_model import *
from PIL import Image
import base64
import io
import os
import joblib
import pickle
# Add more imports if required



###########################################################################################################################################
#         Caution: Don't change any of the filenames, function names and definitions                                                      #
#        Always use the current_path + file_name for refering any files, without it we cannot access files on the server                  # 
###########################################################################################################################################

# Current_path stores absolute path of the file from where it runs. 
current_path = os.path.dirname(os.path.abspath(__file__))

# ---------------- Memory-light model loading (Render free tier = 512MB RAM) ----------------
# 1 CPU thread -> less per-thread memory on a small instance
torch.set_num_threads(1)

_siamese_cache = {}
def _load_siamese(device):
    """Load the Siamese model ONCE with minimal peak memory:
    a) build the network on the 'meta' device (allocates NO weight memory)
    b) memory-map the checkpoint file instead of reading it fully into RAM
    c) assign=True -> use the checkpoint tensors directly (no second copy)
    d) dynamic int8 quantization of Linear layers: fc1 160MB -> ~40MB
    """
    if 'net' not in _siamese_cache:
        import gc
        with torch.device('meta'):
            net = SiameseNetwork()
        try:
            ckpt = torch.load(current_path + '/siamese_model.t7', map_location='cpu', mmap=True)
        except Exception:
            # fallback if the file format does not support mmap
            ckpt = torch.load(current_path + '/siamese_model.t7', map_location='cpu')
        net.load_state_dict(ckpt['net_dict'], assign=True)
        net.float()   # checkpoint is stored as float16 -> back to float32 for CPU inference
        net.eval()
        # inplace=True -> no deepcopy of the model during quantization
        net = torch.ao.quantization.quantize_dynamic(net, {nn.Linear}, dtype=torch.qint8, inplace=True)
        del ckpt
        gc.collect()
        _siamese_cache['net'] = net
    return _siamese_cache['net']

# Classifier (Stage 3) - loaded ONCE. File holds {'classifier': sklearn model, 'classes': [names]}
_clf_cache = {}
def _load_classifier():
    if 'clf' not in _clf_cache:
        obj = joblib.load(current_path + '/face_classifier.sav')
        if isinstance(obj, dict):
            _clf_cache['clf'] = obj['classifier']
            _clf_cache['classes'] = obj.get('classes', classes)
        else:                       # plain sklearn model -> fall back to 'classes' list in face_recognition_model.py
            _clf_cache['clf'] = obj
            _clf_cache['classes'] = classes
    return _clf_cache['clf'], _clf_cache['classes']

#1) The below function is used to detect faces in the given image.
#2) It returns only one image which has maximum area out of all the detected faces in the photo.
#3) If no face is detected,then it returns zero(0).

def detected_face(image):
    eye_haar = current_path + '/haarcascade_eye.xml'
    face_haar = current_path + '/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_haar)
    eye_cascade = cv2.CascadeClassifier(eye_haar)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_areas=[]
    images = []
    required_image=0
    for i, (x,y,w,h) in enumerate(faces):
        face_cropped = gray[y:y+h, x:x+w]
        face_areas.append(w*h)
        images.append(face_cropped)
        required_image = images[np.argmax(face_areas)]
        required_image = Image.fromarray(required_image)
    return required_image


#1) Images captured from mobile is passed as parameter to the below function in the API call. It returns the similarity measure between given images.
#2) The image is passed to the function in base64 encoding, Code for decoding the image is provided within the function.
#3) Define an object to your siamese network here in the function and load the weight from the trained network, set it in evaluation mode.
#4) Get the features for both the faces from the network and return the similarity measure, Euclidean,cosine etc can be it. But choose the Relevant measure.
#5) For loading your model use the current_path+'your model file name', anyhow detailed example is given in comments to the function 
#Caution: Don't change the definition or function name; for loading the model use the current_path for path example is given in comments to the function
def get_similarity(img1, img2):
    device = torch.device("cpu")  # quantized model runs on CPU only
    
    det_img1 = detected_face(img1)
    det_img2 = detected_face(img2)
    # If no face found in an image, fall back to the full grayscale image (checked per image)
    if not isinstance(det_img1, Image.Image):
        det_img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    if not isinstance(det_img2, Image.Image):
        det_img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    face1 = trnscm(det_img1).unsqueeze(0)
    face2 = trnscm(det_img2).unsqueeze(0)
    ##########################################################################################
    ##Example for loading a model using weight state dictionary:                            ##
    ## feature_net = light_cnn() #Example Network                                           ##
    ## model = torch.load(current_path + '/siamese_model.t7', map_location=device)          ##
    ## feature_net.load_state_dict(model['net_dict'])                                       ##
    ##                                                                                      ##
    ##current_path + '/<network_definition>' is path of the saved model if present in       ##
    ##the same path as this file, we recommend to put in the same directory                 ##
    ##########################################################################################
    ##########################################################################################
    
    # Load the model (same architecture as training) on CPU/GPU and set eval mode
    feature_net = _load_siamese(device)

    # Euclidean distance between embeddings = DISSIMILARITY
    # (same measure used in contrastive loss: low = same person, high = different person)
    with torch.no_grad():
        output1, output2 = feature_net(face1.to(device), face2.to(device))
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)

    return round(float(euclidean_distance.item()), 4)
    
#1) Image captured from mobile is passed as parameter to this function in the API call, It returns the face class in the string form ex: "Person1"
#2) The image is passed to the function in base64 encoding, Code to decode the image provided within the function
#3) Define an object to your network here in the function and load the weight from the trained network, set it in evaluation mode
#4) Perform necessary transformations to the input(detected face using the above function).
#5) Along with the siamese, you need the classifier as well, which is to be finetuned with the faces that you are training
##Caution: Don't change the definition or function name; for loading the model use the current_path for path example is given in comments to the function
def get_face_class(img1):
    device = torch.device("cpu")  # quantized model runs on CPU only
    
    det_img1 = detected_face(img1)
    if not isinstance(det_img1, Image.Image):
        det_img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    ##YOUR CODE HERE, return face class here
    ##Hint: you need a classifier finetuned for your classes, it takes o/p of siamese as i/p to it
    ##Better Hint: Siamese experiment is covered in one of the labs
    face1 = trnscm(det_img1).unsqueeze(0)            # same transform as training: 100x100 grayscale tensor
    feature_net = _load_siamese(device)               # same (shared) Siamese model as get_similarity
    with torch.no_grad():
        embedding = feature_net.forward_once(face1.to(device)).numpy()
    clf, class_names = _load_classifier()
    pred = int(clf.predict(embedding)[0])
    return str(class_names[pred])