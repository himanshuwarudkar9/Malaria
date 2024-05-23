import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import os

# Function to download and combine chunks
def download_and_combine_chunks(chunk_urls, output_file):
    with open(output_file, 'wb') as output:
        for url in chunk_urls:
            response = requests.get(url)
            output.write(response.content)

# Define chunk URLs (GitHub raw URLs)
chunk_urls = [
 'https://github.com/himanshuwarudkar9/Malaria/blob/main/malaria_classification_model.zip.part0',
    'https://github.com/himanshuwarudkar9/Malaria/blob/main/malaria_classification_model.zip.part1',
    # Add more parts as necessary
]

# Download and combine chunks if model file doesn't exist
zip_path = 'malaria_classification_model.zip'
if not os.path.exists(zip_path):
    download_and_combine_chunks(chunk_urls, zip_path)

# Unzip the combined file
import zipfile
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall()

# Load the model
model = tf.keras.models.load_model('malaria_classification_model.h5')

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
