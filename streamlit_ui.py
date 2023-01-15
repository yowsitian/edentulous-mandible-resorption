import sys

import streamlit as st

import io
import os
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
from functools import partial
from albumentations.pytorch import ToTensorV2
import albumentations as A
import unet_model as u_model
import boneHeightMeasurement as bhm

# functions

# Loading model from checkpoint
def load_model(model, checkpoint):
    state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
    if "model_state_dict" in state_dict.keys():
        state_dict = state_dict['model_state_dict']
        
    state_dict = {k[7:] if k.startswith('module.') else k : state_dict[k] for k in state_dict.keys()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

# Specify title

st.set_page_config(page_title="Mandible Resorption Pattern Recognition", page_icon="🔗",
                   layout="wide")  # needs to be the first thing after the streamlit import

st.title('Bone Resorption Pattern Recognition of Edentulous Mandible')
st.markdown("Our system allows you to identify the **bone height for anterior and posterior region** of mandible and recognize the **bone resorption severity level** of the respective region from the dental panoramic radiograph images.")
st.subheader('Data Input')

left, right = st.columns([3,2])
with left:
    # Create file uploader
    uploaded_data = st.file_uploader(
    'Please upload your dental panoramic radiograph image here.', 
    type=['png','jpg', 'jpeg'],
    accept_multiple_files=False
   )
    boneHeightImage = Image.open('boneHeightRange2.png')
    st.image(boneHeightImage, caption='Bone height range for each region and severity level from the current database')      

def displayOriImage(d):
    data = d.getvalue()
    # Convert data to bytes
    data = io.BytesIO(data)
    # Convert bytes to image
    data = Image.open(data)
    # Display image
    st.image(data, caption=uploaded_data.name)

# Specify model checkpoint
MODEL_LEFT_CHECKPOINT = 'modelcheckpoint/left_model_checkpoints.pth'
MODEL_ANT_CHECKPOINT = 'modelcheckpoint/ant_model_checkpoints1.pth'
MODEL_RIGHT_CHECKPOINT = 'modelcheckpoint/right_model_checkpoints.pth'
MODEL_CHECKPOINT_UNET = 'modelcheckpoint/model_checkpoints_unet.pth'

AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
AlexNet_model.classifier[4] = nn.Linear(4096,1024)
AlexNet_model.classifier[6] = nn.Linear(1024,3)

# Load model
AlexNet_model_ant = load_model(AlexNet_model, MODEL_ANT_CHECKPOINT)
AlexNet_model_left = load_model(AlexNet_model, MODEL_LEFT_CHECKPOINT)
AlexNet_model_right = load_model(AlexNet_model, MODEL_RIGHT_CHECKPOINT)
model_unet = load_model(u_model.unet_model(), MODEL_CHECKPOINT_UNET)

def save_file(uploadedfile, name):
     with open(os.path.join("tempDir",name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return os.path.join("tempDir",name)

LABEL_TO_COLOR = {0:[0,0,0], 1:[128,0,0], 2:[0,128,0], 3:[128,128,0], 4:[0,0,128], 5:[128,0,128], 6:[0,128,128], 7:[128,128,128], 8:[64,0,0], 9:[192,0,0]}

def mask2rgb(mask):
    rgb = np.zeros(mask.shape+(3,), dtype=np.uint8)
    for i in np.unique(mask):
            rgb[mask==i] = LABEL_TO_COLOR[i]
    return rgb

# Check file is present or not
if uploaded_data is not None:
    st.subheader("Bone Resorption Severity Level Result")
    st.markdown("Input Image")
    imagePath = save_file(uploaded_data, uploaded_data.name)
    displayOriImage(uploaded_data)

else:
    st.subheader("Sample Bone Resorption Severity Level Result")
    st.markdown("Original Image")
    image = Image.open('test.JPG')
    st.image(image, caption='Image input')     

# Specify GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_ant = AlexNet_model_ant.to(DEVICE)
model_left = AlexNet_model_left.to(DEVICE)
model_right = AlexNet_model_right.to(DEVICE)
model_unet = model_unet.to(DEVICE)

tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

t1 = A.Compose([
    A.Resize(160,240),
    A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

classList = ["Mild","Moderate","Severe"]

def outToClass(out):
    return classList[out.item()]

softmax = nn.Softmax(dim=1)

if uploaded_data is not None:
    
    open_cv_image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    height, width = open_cv_image.shape[0], open_cv_image.shape[1]
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2RGB)

    t2 = A.Compose([A.Resize(height,width)])
    
    # bone severity classification
    tf_data = tf(open_cv_image)
    tf_data = tf_data.to(DEVICE).unsqueeze(0)

    # bone segmentation
    unet_data = t1(image=open_cv_image)['image']
    unet_data = unet_data.to(DEVICE).unsqueeze(0)

    with torch.no_grad():
       output_ant = model_ant(tf_data)
       _, predicted_ant = torch.max(output_ant, 1)

       output_left = model_left(tf_data)
       _, predicted_left = torch.max(output_left, 1)

       output_right = model_right(tf_data)
       _, predicted_right = torch.max(output_right, 1)

       preds = torch.argmax(softmax(model_unet(unet_data)),axis=1).to('cpu')
       preds1 = np.array(preds[0,:,:])

       rgb_final_mask = mask2rgb(preds1)

       # Convert image to bytes
       pil_im = Image.fromarray(rgb_final_mask)


       b = io.BytesIO()
       pil_im.save(b, 'jpeg')
       im_bytes = b.getvalue()

       final_mask = io.BytesIO(im_bytes)
       # Convert bytes to image
       pred_data = Image.open(final_mask)
       # Display image
       st.image(pred_data, caption="Predicted Mask")

       rgb_final_mask= t2(image=rgb_final_mask)['image']

       boneHeightC, boneHeightL, boneHeightR = bhm.getBoneHeightWithImage(imagePath, rgb_final_mask)
    
    st.write(f'boneHeightC {boneHeightC}, boneHeightL: {boneHeightL}, boneHeightR: {boneHeightR}')



    st.write(f'Prediction (Right Posterior): {outToClass(predicted_right)}')
    st.write(f'Prediction (Anterior): {outToClass(predicted_ant)}')
    st.write(f'Prediction (Left Posterior): {outToClass(predicted_left)}')       


