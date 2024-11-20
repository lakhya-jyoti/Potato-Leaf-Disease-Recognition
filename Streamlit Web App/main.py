#importing Libraries
import streamlit as st
import tensorflow as tf
import numpy as np
import time
from PIL import Image
import io

# Model prediction function
def model_prediction(image_data):
    model = tf.keras.models.load_model('Models/potato_accuracy95.keras')
    image = Image.open(image_data).resize((128, 128))  # Open and resize the image
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

#sidebar

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("", ["Home", "About","Disease Recognition"])


#Home page

if(app_mode == "Home"):
    st.markdown("<h2 style='color: #1ABC9C;'>POTATO LEAF PLANT DISEASE RECOGNITION SYSTEM</h2>", unsafe_allow_html=True)
    image_path = "images/index-banner.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown(""" ### Welcome to the **Potato Leaf Disease Recognition System**

Our mission is to assist in identifying Potato plant diseases quickly and accurately. Simply upload an image, and our system will automatically detect the disease.

### How It Works:

- **Upload Image** – Submit a photo of the plant.
- **Analysis** – The system processes the image to identify potential diseases.
- **Result** – Instantly receive the disease name.

### Why Choose Us:

- High Accuracy
- User-Friendly Experience
- Time-Saving and Efficient
                """)
    
#about page

elif(app_mode=="About"):
    st.markdown("<h2 style='color: #1ABC9C;'>ABOUT</h2>", unsafe_allow_html=True)
    st.markdown("""
    ### About Dataset
    Original Dataset is available here: https://www.kaggle.com/datasets/muhammadardiputra/potato-leaf-disease-dataset
    .The dataset is recreated using offline augmentation from the original dataset.
    
    ## Content
    1. Train (1722 images)
    2. Valid (215 images)
    3. Test (215 images)
    4. 3 classes: Early Blight, Late Blight, Healthy            
                """)
    
#prediction page

elif(app_mode == "Disease Recognition"):
    st.markdown("<h2 style='color: #1ABC9C;'>Upload Image or Capture</h2>", unsafe_allow_html=True)

    option = st.radio(" ", ('Upload Image', 'Capture Image'))
    test_image = None
    
    if option == 'Upload Image':
        test_image = st.file_uploader("Choose an image:")
        
    # Camera input option
    if option == 'Capture Image':
        captured_image = st.camera_input("Capture an image")
        if captured_image:
            test_image = io.BytesIO(captured_image.getvalue()) # Convert to BytesIO for PIL compatibility
    
    
    # Check if an image has been provided
    if test_image is not None:
        # Display the "Show Image" button
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)
        # Display the "Predict" button
        if st.button("Predict"):
            with st.spinner("Please Wait..."):
                time.sleep(1)
                st.write("### Our Prediction")
                result_index = model_prediction(test_image)
                # Define class names (adjust according to your labels)
                class_name = ['Potato__Late_blight', 'Potato__Early_blight', 'Potato__Healthy']
                st.success(f"Model is predicting it is a {class_name[result_index]}")       

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color:#1ABC9C;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by <a style='display: block; text-align: center;' href="https://www.linkedin.com/in/lakhya-borah-2a90b8206/" target="_blank">Lakhya Borah</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)