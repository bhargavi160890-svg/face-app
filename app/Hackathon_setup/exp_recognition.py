import numpy as np
import cv2
# matplotlib removed (not used on the server, saves RAM)
import torch
# In the below line,remove '.' while working on your local system.However Make sure that '.' is present before face_recognition_model while uploading to the server, Do not remove it.
from .exp_recognition_model import *
from PIL import Image
import base64
import io
import os
## Add more imports if required

#############################################################################################################################
#   Caution: Don't change any of the filenames, function names and definitions                                              #
#   Always use the current_path + file_name for refering any files, without it we cannot access files on the server         #
#############################################################################################################################

# Current_path stores absolute path of the file from where it runs.
current_path = os.path.dirname(os.path.abspath(__file__))

# Load the expression model ONCE (not on every request). File holds {'net_dict': float16 weights, 'classes': [names]}
_exp_cache = {}
def _load_expression_model():
    if 'net' not in _exp_cache:
        ckpt = torch.load(current_path + '/expression_model.t7', map_location='cpu')
        _exp_cache['tta_flag'] = ckpt.get('tta', False)   # test-time augmentation (image + mirror), same as Colab eval
        names = ckpt.get('classes') or [classes[i] for i in sorted(classes)]
        if 'arch' in ckpt:   # v3: pretrained backbone (efficientnet_b0 / mobilenet_v3_large) fine-tuned in Colab
            net = build_tl_model(ckpt['arch'], num_classes=len(names), pretrained=False)
            tf = make_tl_transform(ckpt.get('img_size', 112), ckpt.get('mean', (0.485, 0.456, 0.406)),
                                   ckpt.get('std', (0.229, 0.224, 0.225)))
        else:                # v1/v2: own CNN
            net = facExpRec(num_classes=len(names), **ckpt.get('config', {}))   # same widths/drops as training
            tf = make_transform(ckpt.get('img_size', 48))
        net.load_state_dict(ckpt['net_dict'])   # float16 -> float32 copied automatically
        net.float().eval()
        del ckpt
        _exp_cache['net'], _exp_cache['classes'], _exp_cache['tf'] = net, names, tf
        _exp_cache['tta'] = bool(_exp_cache.get('tta_flag', False))
    return _exp_cache['net'], _exp_cache['classes'], _exp_cache['tf']


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
    

#1) Images captured from mobile is passed as parameter to the below function in the API call, It returns the Expression detected by your network.
#2) The image is passed to the function in base64 encoding, Code for decoding the image is provided within the function.
#3) Define an object to your network here in the function and load the weight from the trained network, set it in evaluation mode.
#4) Perform necessary transformations to the input(detected face using the above function), this should return the Expression in string form ex: "Anger"
#5) For loading your model use the current_path+'your model file name', anyhow detailed example is given in comments to the function 
##Caution: Don't change the definition or function name; for loading the model use the current_path for path example is given in comments to the function
def get_expression(img):
    device = torch.device("cpu")
    
    ##########################################################################################
    ##Example for loading a model using weight state dictionary:                            ##
    ## face_det_net = facExpRec() #Example Network                                          ##
    ## model = torch.load(current_path + '/exp_recognition_net.t7', map_location=device)    ##
    ## face_det_net.load_state_dict(model['net_dict'])                                      ##
    ##                                                                                      ##
    ##current_path + '/<network_definition>' is path of the saved model if present in       ##
    ##the same path as this file, we recommend to put in the same directory                 ##
    ##########################################################################################
    ##########################################################################################

    face = detected_face(img)
    if not isinstance(face, Image.Image):      # no face found -> use full grayscale image
        face = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    
    # YOUR CODE HERE, return expression using your model
    net, names, tf = _load_expression_model()
    x = tf(face).unsqueeze(0).to(device)        # gray -> resize -> tensor (same as training eval transform)
    with torch.no_grad():
        logits = net(x)
        if _exp_cache.get('tta'):
            logits = logits + net(torch.flip(x, dims=[3]))   # mirrored face -> more stable prediction
        pred = int(logits.argmax(1).item())
    return str(names[pred])