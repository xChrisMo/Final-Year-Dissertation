import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.losses import BinaryCrossentropy
import tempfile

st.title("Rail Track App Test")

@st.cache_resource
def deserialize_binary_crossentropy(config):
    if 'reduction' in config:
        config.pop('reduction')
    return BinaryCrossentropy.from_config(config)

def load_model_cached():
    return load_model('vgg109.h5', custom_objects={'BinaryCrossentropy': deserialize_binary_crossentropy})

model = load_model_cached()

def load_and_preprocess_image(img_path):
    # Load the image file, resizing it to the input size of our model
    img = image.load_img(img_path, target_size=(256,256))

    # Convert the image to array
    img_array = image.img_to_array(img)

    # Normalize the image pixels
    img_array /= 255.0

    # Expand dimensions to fit the batch size
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict_image(img_path):
    # Preprocess the image
    img_array = load_and_preprocess_image(img_path)

    # Make predictions
    predictions = model.predict(img_array)

    # Decode the predictions to the original labels
    predicted_label = "LEAVES" if predictions[0] > 0 else "Clean"

    return predicted_label

def process_uploaded_image(uploaded_file):
    """Processes an uploaded image and makes a prediction."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(uploaded_file.getbuffer())

        # Display the uploaded image
        st.image(temp_file_path, caption='Uploaded Image.', use_column_width=True)

        # Make a prediction
        st.write("Classifying...")
        label = predict_image(temp_file_path)  # Use the temporary file path
        st.write(f"Prediction: {label}")

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    process_uploaded_image(uploaded_file)