# Filename: app_small.py

import streamlit as st
import matplotlib.pyplot as plt
import cv2

# Image path
path = "dog.jpg"

st.title("Web app using Streamlit")

# Display image
fig = plt.figure(figsize=(3,3))
plt.imshow(cv2.imread(path)[:,:,::-1])
plt.axis("off")
st.pyplot(dpi=300,fig=fig,clear_figure=True)