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
@st.cache
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

fileList = ["my_plot_left.png","my_plot_right.png", "my_plot_center.png"]
for i in fileList:
    if os.path.exists(i):
        os.remove(i)

LABEL_TO_COLOR = {0:[0,0,0], 1:[128,0,0], 2:[0,128,0], 3:[128,128,0], 4:[0,0,128], 5:[128,0,128], 6:[0,128,128], 7:[128,128,128], 8:[64,0,0], 9:[192,0,0]}

st.write(
    """
    <style>
    [data-testid="stMetricDelta"] svg {
        display: none;
    }
    [data-testid="stMetricLabel"] {
        font-weight: bold;
    }
    [data-testid="stFileUploader"] {
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
uploaded_data = None
left, right = st.columns([3,2])
with left:
    # Create file uploader
    uploaded_data = st.file_uploader(
    'Upload your dental panoramic radiograph image', 
    type=['png','jpg', 'jpeg'],
    accept_multiple_files=False
    )
    # with open("sample_image_input.JPG", "rb") as file:
    #     btn = st.download_button(
    #         label="Download sample image for data testing",
    #         data=file,
    #         file_name="sample_image_input.JPG",
    #         mime="image/png"
    #     )
    boneHeightImage = Image.open('boneHeightRange2.png')
    st.image(boneHeightImage, caption='Bone height range for each region and severity level from the current database')   

with right:
    st.text(' ')
    st.text(' ')
    with st.expander("Download sample dental panoramic radiograph images for data input testing"):
        with open("sample_image_input.JPG", "rb") as file1:
            btn1 = st.download_button(
                label="Sample Image 1",
                data=file1,
                file_name="sample_image_input_1.JPG",
                mime="image/jpg",
                key = "Sample Image 1"
            )
        with open("sample_image_input_2.JPG", "rb") as file2:
            btn2 = st.download_button(
                label="Sample Image 2",
                data=file2,
                file_name="sample_image_input_2.JPG",
                mime="image/jpg",
                key = "Sample Image 2"
            )
        with open("sample_image_input_3.JPG", "rb") as file3:
            btn3 = st.download_button(
                label="Sample Image 3",
                data=file3,
                file_name="sample_image_input_3.JPG",
                mime="image/jpg",
                key = "Sample Image 3"
            )
        with open("sample_image_input_4.JPG", "rb") as file4:
            btn4 = st.download_button(
                label="Sample Image 4",
                data=file4,
                file_name="sample_image_input_4.JPG",
                mime="image/jpg",
                key = "Sample Image 4"
            )
        with open("sample_image_input_5.JPG", "rb") as file5:
            btn4 = st.download_button(
                label="Sample Image 5",
                data=file5,
                file_name="sample_image_input_5.JPG",
                mime="image/jpg",
                key = "Sample Image 5"
            )

def displayOriImage(d):
    data = d.getvalue()
    # Convert data to bytes
    data = io.BytesIO(data)
    # Convert bytes to image
    data = Image.open(data)
    # Display image
    st.image(data, caption="Image Input: "+ uploaded_data.name)

# # Specify model checkpoint
# MODEL_LEFT_CHECKPOINT = 'modelcheckpoint/left_model_checkpoints.pth'
# MODEL_ANT_CHECKPOINT = 'modelcheckpoint/ant_model_checkpoints1.pth'
# MODEL_RIGHT_CHECKPOINT = 'modelcheckpoint/right_model_checkpoints.pth'
MODEL_CHECKPOINT_UNET = 'modelcheckpoint/model_checkpoints_unet.pth'

# AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
# AlexNet_model.classifier[4] = nn.Linear(4096,1024)
# AlexNet_model.classifier[6] = nn.Linear(1024,3)

# Load model
# AlexNet_model_ant = load_model(AlexNet_model, MODEL_ANT_CHECKPOINT)
# AlexNet_model_left = load_model(AlexNet_model, MODEL_LEFT_CHECKPOINT)
# AlexNet_model_right = load_model(AlexNet_model, MODEL_RIGHT_CHECKPOINT)
model_unet = load_model(u_model.unet_model(), MODEL_CHECKPOINT_UNET)

def save_file(uploadedfile):
     with open(os.path.join("tempDir","input.png"),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return os.path.join("tempDir","input.png")

def mask2rgb(mask):
    rgb = np.zeros(mask.shape+(3,), dtype=np.uint8)
    for i in np.unique(mask):
            rgb[mask==i] = LABEL_TO_COLOR[i]
    return rgb

# Check file is present or not
if uploaded_data is not None:
    st.subheader("Bone Resorption Severity Level Result")

# Specify GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model_ant = AlexNet_model_ant.to(DEVICE)
# model_left = AlexNet_model_left.to(DEVICE)
# model_right = AlexNet_model_right.to(DEVICE)
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

def getEstSev(h, region):
    if(h == 0):
        return ["Region not found"]
    if(region == "r"):
        if(h > 5.75):
            return ["Mild"]

        # Mild to moderate
        if(h > 5.38):
            return ["Mild"]
        # Mild to moderate
        if(h >= 5.01):
            return ["Moderate"]

        if(h > 4.44 and h < 5.01):
            return ["Moderate"]
        
        # Moderate to severe
        if(h > 3.94):
            return ["Moderate"]
        # Moderate to severe
        if(h >= 3.43):
            return ["Severe"]
        
        if(h < 3.43):
            return ["Severe"]
    if(region == "c"):
        if(h > 6.51):
            return ["Mild"]

        # Mild to moderate
        if(h > 5.72):
            return ["Mild"]
        # Mild to moderate
        if(h >= 4.92):
            return ["Moderate"]

        if(h > 4.37 and h < 4.92):
            return ["Moderate"]
        
        # Moderate to severe
        if(h > 4.01):
            return ["Moderate"]
        # Moderate to severe
        if(h >= 3.65):
            return ["Severe"]

        if(h < 3.65):
            return ["Severe"]
    if(region == "l"):
        if(h > 6.88):
            return ["Mild"]

        # Mild to moderate
        if(h > 5.95):
            return ["Mild"]
        # Mild to moderate
        if(h >= 5.02):
            return ["Moderate"]

        if(h > 4.33 and h < 5.02):
            return ["Moderate"]
            
         # Moderate to severe
        if(h > 3.74):
            return ["Moderate"]
        # Moderate to severe
        if(h >= 3.15):
            return ["Severe"]

        if(h < 3.15):
            return ["Severe"]


softmax = nn.Softmax(dim=1)

def regionIsValid(path, h):
    if(h == 0):
        return False
    if(os.path.exists(path)):
        return True
    return False

if uploaded_data is not None:
    with st.spinner('Loading...'):
        imagePath = save_file(uploaded_data)     
        ori_input_image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

        ori_input_image = ori_input_image[25:556, 245:1295]

        height, width = ori_input_image.shape[0], ori_input_image.shape[1]
        open_cv_image = cv2.cvtColor(ori_input_image, cv2.COLOR_GRAY2RGB)

        t2 = A.Compose([A.Resize(969,1908)])
    
        # bone severity classification
        tf_data = tf(open_cv_image)
        tf_data = tf_data.to(DEVICE).unsqueeze(0)

        # bone segmentation
        unet_data = t1(image=open_cv_image)['image']
        unet_data = unet_data.to(DEVICE).unsqueeze(0)

        with torch.no_grad():
        #    output_ant = model_ant(tf_data)
        #    _, predicted_ant = torch.max(output_ant, 1)

        #    output_left = model_left(tf_data)
        #    _, predicted_left = torch.max(output_left, 1)

        #    output_right = model_right(tf_data)
        #    _, predicted_right = torch.max(output_right, 1)

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

            rgb_final_mask= t2(image=rgb_final_mask)['image']
            final_input_image = t2(image=ori_input_image)['image']

            boneHeightC, boneHeightL, boneHeightR = bhm.getBoneHeightWithImage(final_input_image, rgb_final_mask)

            estSevCR = getEstSev(boneHeightR, "r")
            estSevCC = getEstSev(boneHeightC, "c")
            estSevCL = getEstSev(boneHeightL, "l")

        st.write("#")
        st.markdown(
        """
        <style>
        div[data-testid="column"]:nth-of-type(1)
        {
            text-align: center;
        } 

        div[data-testid="column"]:nth-of-type(2)
        {
            text-align: center;
        } 
        div[data-testid="column"]:nth-of-type(3)
        {
            text-align: center;
        } 
        </style>
        """,unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1,1,1])

        bR = regionIsValid('my_plot_right.png', boneHeightR)
        bC = regionIsValid('my_plot_center.png', boneHeightC)
        bL = regionIsValid('my_plot_left.png', boneHeightL)

        with col1:
            if bR:
                rightImage = Image.open('my_plot_right.png')
                st.image(rightImage, caption=None)      
    
        with col2:
            if bC:
                centerImage = Image.open('my_plot_center.png')
                st.image(centerImage, caption=None)   
    
        with col3:
            if bL:
                leftImage = Image.open('my_plot_left.png')
                st.image(leftImage, caption=None)   
        
        colA, colB, colC = st.columns([1,1,1])
        with colA:
            if bR: 
                st.metric("Right Posterior Mandible", estSevCR[0], f"Bone Height: {(boneHeightR*0.5417774472783):.2f} cm")
            else:
                st.metric("Right Posterior Mandible", "Region not found")
    
        with colB:
            if bC:
                st.metric("Anterior Mandible", estSevCC[0], f"Bone Height: {(boneHeightC*0.5417774472783):.2f} cm")
            else:
                st.metric("Anterior Mandible", "Region not found")
    
        with colC:
            if bL:
                st.metric("Left Posterior Mandible", estSevCL[0], f"Bone Height: {(boneHeightL*0.5417774472783):.2f} cm") 
            else:
                st.metric("Left Posterior Mandible", "Region not found") 
    
        st.write("#")
        displayOriImage(uploaded_data)    

