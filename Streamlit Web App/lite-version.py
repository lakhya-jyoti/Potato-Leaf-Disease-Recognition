#Here Tensorflow lite Version is used

#importing Libraries
import streamlit as st
import numpy as np
import time
from PIL import Image
import io
import tensorflow as tf

# Load TFLite model and allocate tensors
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path='Models/potato_accuracy95.tflite')
    interpreter.allocate_tensors()
    return interpreter

# Perform inference on the image using TFLite model
def tflite_model_prediction(interpreter, image_data):
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the image
    image = Image.open(image_data).resize((128, 128))  # Resize the image
    input_arr = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)  # Normalize and add batch dimension

    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], input_arr)

    # Run the model
    interpreter.invoke()

    # Get the results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    result_index = np.argmax(output_data)
    return result_index

# Title for the app
st.title("Potato Disease Detection")

# Options for capturing or uploading an image
st.header("Upload an image or Capture")
option = st.radio(" ", ('Upload Image', 'Capture Image'))

test_image = None

# File uploader option
if option == 'Upload Image':
    test_image = st.file_uploader("Choose an image:")

# Camera input option
if option == 'Capture Image':
    captured_image = st.camera_input("Capture an image")
    if captured_image:
        test_image = io.BytesIO(captured_image.getvalue())  # Convert to BytesIO for PIL compatibility

# Check if an image has been provided
if test_image is not None:
    # Display the "Show Image" button
    if st.button("Show Image"):
        st.image(test_image, use_column_width=True)

    # Display the "Predict" button
    if st.button("Predict"):
        with st.spinner("Please Wait..."):
            time.sleep(3)
            st.write("### Our Prediction")
            
            # Load the TFLite model and perform prediction
            interpreter = load_tflite_model()
            result_index = tflite_model_prediction(interpreter, test_image)

            # Define class names (adjust according to your labels)
            class_name = ['Potato__Late_blight', 'Potato__Early_blight', 'Potato__Healthy']
            st.success(f"Model is predicting it is a {class_name[result_index]}")
