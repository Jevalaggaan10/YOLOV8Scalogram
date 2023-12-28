import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import os


def classify_scalogram(image,modelname):
    # Simulate a time-consuming task
    time.sleep(3)
    model = load_model(modelname, custom_objects={'KerasLayer': hub.KerasLayer})
    dec = {0: 'Normal', 1: 'Interictal'}
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # Reshape the image for prediction
    img = img / 255

    # Reshape and normalize the image
    image_reshaped = np.reshape(img, [1, 224, 224, 3])

    # Make prediction using the transfer learning model
    prediction = model.predict(image_reshaped)

    # Get the predicted label
    predicted_label = np.argmax(prediction)
    confidence_score = prediction[0][predicted_label]
    print(predicted_label)
    return f"{dec[predicted_label]} (Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%)"  # Rep


st.header("Scalogram Epilepsy Classifier")

uploaded_file = st.file_uploader("Upload a Scalogram", type=["png"])
if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".png"):
        st.success("Uploaded file is a PNG image!")
        # image_bytes = uploaded_file.read()
        # image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        # Display the image using Streamlit
        # st.image(image_array, caption="Uploaded PNG Image", use_column_width=True)
        # print(uploaded_file)
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded PNG Image", use_column_width=True)
        if st.button(label="Classify Scalogram", type="primary"):
            with st.spinner("Classifying..."):
                result = classify_scalogram(image,'mobile_netv2_Scalogram_without_COI_preprocessed.keras')
                st.success(f"Classification result: {result}")
    else:
        st.warning("Please upload a valid PNG file.")
