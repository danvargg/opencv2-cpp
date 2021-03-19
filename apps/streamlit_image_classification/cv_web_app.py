# Filename: cv_web_app.py

import streamlit as st
import matplotlib.pyplot as plt
from utils import *

st.title("Image Classification")

# Disable warning
st.set_option('deprecation.showfileUploaderEncoding', False)

# ONNX model file path
model = "model.onnx"

# Convert model to ONNX
convert_model()

# Create a button for loading an image
image_path = st.file_uploader("",type=["png", "jpg", "jpeg"])

if image_path is not None:
    # Read image
    image = read_image(image_path)
    
    # Carry inference
    class_name = model_inference(model,image)  

    # Display image
    st.image(image, use_column_width=True)

    st.title("Class recognized: {}".format(class_name))