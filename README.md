# Malaria Cell Classification

This project involves building a deep learning model to classify malaria-infected cells using Convolutional Neural Networks (CNN). The model is trained on the publicly available dataset and deployed using Streamlit for user interaction.

## Live Demo

You can access the live Streamlit application here: [Malaria Cell Classification App](https://malaria-elspvymyt7ulxtkf88udk5.streamlit.app/)

## Project Overview

Malaria is a life-threatening disease caused by parasites that are transmitted to people through the bites of infected female Anopheles mosquitoes. Early and accurate diagnosis is crucial for effective treatment and management of the disease. This project aims to aid in the diagnosis process by leveraging deep learning techniques to classify cells as either 'Parasitized' or 'Uninfected'.

## Dataset

The dataset used for training and validation is the [Cell Images for Malaria Detection](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria). It contains 27,558 images categorized into 'Parasitized' and 'Uninfected'.

## Model Architecture

A Convolutional Neural Network (CNN) is utilized for this classification task. The model consists of several convolutional layers followed by max-pooling layers, a fully connected dense layer, and a final output layer with a sigmoid activation function.

## Challenges Faced

1. **Computational Resources**:
   - Training deeper networks like ResNet50 and VGG19 was computationally intensive and time-consuming. Hence, the final model selection was based on a balance between accuracy and computational efficiency.

2. **Memory Usage**:
   - Initially, larger image sizes (150x150) and batch sizes were used, which led to memory issues. Reducing the image size to 128x128 and the batch size to 16 helped mitigate this problem.

3. **Training Time**:
   - The training process was time-consuming, especially with a higher number of epochs. Reducing the epochs from 10 to 2 helped in faster training while maintaining a reasonable accuracy.

4. **Model Saving**:
   - Saving models like ResNet50 resulted in large file sizes (>25MB), making them impractical for deployment. The final model was quantized to reduce the size to 2.5MB, making it suitable for deployment on platforms like Streamlit.

## Model Performance

The final CNN model achieved:
- **Training Accuracy**: 92.28%
- **Validation Accuracy**: 92.18%

## How to Use

### Prerequisites

- Python 3.6+
- TensorFlow
- Streamlit
- Pillow
- Numpy
- Matplotlib

### Running the Streamlit App

1. Place the quantized model (`quantized_malaria_classification_model.tflite`) in the project directory.
2. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

### Using the App

1. Upload a cell image (JPEG, JPG, PNG).
2. The app will display the uploaded image.
3. The model will classify the image as 'Parasitized' or 'Uninfected'.
4. Additional information and a disclaimer are provided with the prediction.

## Conclusion

While more complex models like ResNet50 and VGG19 were considered, the CNN model provided a good balance of performance and efficiency. The final deployed model effectively classifies cell images with high accuracy, and the streamlined deployment using Streamlit makes it accessible for educational purposes.

## Disclaimer

This application is intended for educational purposes only. The model's predictions should not be considered definitive medical advice. Always consult a healthcare professional for accurate diagnosis and treatment.

## Acknowledgements

- [Kaggle Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) for providing the data.
- TensorFlow and Streamlit for their amazing libraries and tools.
