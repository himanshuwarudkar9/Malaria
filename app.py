import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the quantized model
interpreter = tf.lite.Interpreter(model_path="quantized_malaria_classification_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define a function to preprocess the image
def preprocess_image(image):
    img = np.array(image.resize((128, 128))) / 255.0  # Resize image to match model input shape and normalize
    return img.reshape(1, 128, 128, 3).astype(np.float32)  # Add batch dimension and convert to float32

# Add custom CSS for background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://research.uga.edu/news/wp-content/uploads/sites/19/2023/04/plasmodium_vivax.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("Malaria Cell Classification")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    input_data = preprocess_image(image)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Display the prediction
    prediction = "Parasitized" if output_data[0][0] < 0.5 else "Uninfected"
    st.write(f"Prediction: {prediction}")
