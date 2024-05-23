import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# Function to download the model if not already downloaded
def download_model():
    url = 'https://drive.google.com/file/d/1zSPCTCc6X2Q5imzpGaNVf67IOckTq7Lc/view?usp=drive_link'  # Replace with your file ID
    output = 'malaria_classification_model.h5'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return output

# Load the model
model_path = download_model()
model = tf.keras.models.load_model(model_path)

# Define a function to preprocess the image
def preprocess_image(image):
    img = np.array(image.resize((150, 150))) / 255.0  # Resize image to match model input shape and normalize
    return img.reshape(1, 150, 150, 3)  # Add batch dimension

# Streamlit UI
st.title("Malaria Cell Classification")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image and make prediction
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)

    # Display the prediction
    if prediction[0] > 0.5:
        st.write("Prediction: Parasitized")
    else:
        st.write("Prediction: Uninfected")
