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

# Streamlit UI
st.title("Malaria Cell Classification")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.subheader("Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    input_data = preprocess_image(image)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Display prediction
    st.subheader("Prediction")
    if output_data[0][0] < 0.5:
        st.success("Prediction: Uninfected")
    else:
        st.error("Prediction: Parasitized")

    # Display confidence score
    confidence_score = output_data[0][0] if output_data[0][0] < 0.5 else 1.0 - output_data[0][0]
    st.write(f"Confidence Score: {confidence_score:.2f}")

    # Display probability distribution
    st.subheader("Probability Distribution")
    st.bar_chart({"Uninfected": 1.0 - confidence_score, "Parasitized": confidence_score})
