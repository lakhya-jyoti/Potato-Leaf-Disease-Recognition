from fastapi import FastAPI, UploadFile, File
import uvicorn
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
from io import BytesIO

# Input shape and class labels (same as Streamlit)
input_shape = (128, 128)
class_labels = ["Potato__Late_blight", "Potato__Early_blight", "Potato__Healthy"]

# Function to read the image (matches Streamlit logic)
def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image

# Function to preprocess the image (matches Streamlit logic)
def preprocess(image: Image.Image):
    # Resize the image
    image = image.resize(input_shape)
    # Convert to numpy array
    input_arr = img_to_array(image)
    # Add batch dimension
    input_arr = np.expand_dims(input_arr, axis=0)
    return input_arr

# Function to load the Keras model
def load_model_file():
    # Ensure the same model file is used
    return load_model("app/Model/potato_accuracy95.keras")

# Function to make predictions
def predict(model, image):
    # Make the prediction
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions)  # Get the highest probability class
    predicted_class_label = class_labels[predicted_class_index]

    return {
        "class": predicted_class_label,
        "raw_predictions": predictions.tolist()
    }

# FastAPI app setup
app = FastAPI()

# Load the model when the server starts
model = load_model_file()

@app.get("/")
def index():
    return {"message": "Welcome to the Potato Disease Detection API!"}

@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_data = await file.read()
        pil_image = read_image(image_data)

        # Preprocess the image
        processed_image = preprocess(pil_image)

        # Make the prediction
        prediction_result = predict(model, processed_image)

        return {"prediction": prediction_result}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, port=8085, host="localhost")
